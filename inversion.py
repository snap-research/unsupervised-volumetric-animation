"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
from modules.util import make_coordinate_grid
from model import UVASystem
from argparse import ArgumentParser
import yaml
import imageio as io
from skimage.transform import resize
from logger import VisualizationCallback
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

def void_logger(*args, **kwargs):
    """
    Send the logs to nowhere.
    """
    return None


class Transform:
    """
    Random euclidean transformation, for perturbing the image in the second stage.
    """

    def __init__(self, bs, sigma_affine, vflip=False):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))

        rotation = torch.cat([
            torch.cos(noise[:, :1, 0]), -torch.sin(noise[:, :1, 0]),
            torch.sin(noise[:, :1, 0]),
            torch.cos(noise[:, :1, 0])
        ],
                             axis=1)
        rotation = rotation.view(bs, 2, 2)
        if vflip:
            # Genrerate a vector of random -1 or 1
            flip = torch.randint(0, 2, [bs, 1, 1]) * 2 - 1
            flip = flip.float()
            # Create a 2x2 flip matrix
            flip = torch.cat([flip, torch.zeros([bs, 1, 1]), torch.zeros([bs, 1, 1]), torch.ones([bs, 1, 1])], axis=1)
            flip = flip.view(bs, 2, 2)
            # Apply the flip
            rotation = rotation @ flip
        self.theta = torch.cat([rotation, noise[:, :, 2:]], axis=-1)

        self.bs = bs

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], device=frame.device).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="zeros", align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)
        return transformed

def init_model(model):
    """
    Initialize the model, assume that target embedding is at index 0.
    And replace it with the mean of all embeddings.
    """
    weight = model.state_dict()['canonical_generator.embedding.embedding.weight']
    weight.data[0] = weight.mean(axis=0)
    model.train()
    model.canonical_generator.log = void_logger
    model.pnp_pose.log = void_logger
    model.log = void_logger
    model.pnp_pose.eval()
    model.canonical_generator.noise_std_current = 0.


def inversion(model, image_batch, inversion_params):
    """
    Perform the PTI inversion given the one or several images.
    PTI inversion is a two-stage process:
        1. Optimize only the embedding to match the image.
        2. Optimize the rest of the parameters to match the image.

    :param model: the UVASystem model
    :param image_batch: the image batch[num_images, 3, h, w]
    :param inversion_params: the hyper-parameters for the inversion dict consisting of:
        lr_first: the learning rate for the first stage
        steps_first: the number of steps for the first stage
        sigma_first: the sigma for the random affine transformation in the first stage
        l1_first: the l1 loss for the first stage 
        use_lr_schedule_first: whether to reduce the learning rate in the first stage
        vflip_first: whether to flip the image vertically in the first stage
        lr_second: the learning rate for the second stage
        steps_second: the number of steps for the second stage
        sigma_second: the sigma for the random affine transformation in the second stage
        l1_second: the l1 loss for the second stage
        reduce_lr_second: whether to reduce the learning rate in the second stage
        vflip_second: whether to flip the image vertically in the second stage
        bs: the batch size to use in case all images do not fit in the GPU, None if all images used
        preserve_geometry_weight: the weight for the geometry preservation loss (for the second stage)
    :param tb: the tensorboard logger, None if not used.
    :return: the latent code
    """
    init_model(model)

    inversion_stage(model,
                    image_batch,
                    video_id=None,
                    only_embedding=True,
                    preserve_geometry_weight=0,
                    lr=inversion_params['lr_first'],
                    steps=inversion_params['steps_first'],
                    bs=inversion_params['bs'],
                    l1_weight=inversion_params['l1_first'], 
                    sigma_affine=inversion_params['sigma_first'],
                    use_lr_schedule=inversion_params['use_lr_schedule_first'],
                    vflip=inversion_params['vflip_first'])

    inversion_stage(model,
                    image_batch,
                    video_id=None,
                    only_embedding=False,
                    preserve_geometry_weight=inversion_params['preserve_geometry_weight'],
                    lr=inversion_params['lr_second'],
                    steps=inversion_params['steps_second'],
                    bs=inversion_params['bs'],
                    l1_weight=inversion_params['l1_second'], 
                    sigma_affine=inversion_params['sigma_second'],
                    use_lr_schedule=inversion_params['use_lr_schedule_second'],
                    vflip=inversion_params['vflip_second'])


def inversion_stage(model,
                    image_batch,
                    video_id=None,
                    only_embedding=True,
                    preserve_geometry_weight=0,
                    lr=1e-4,
                    steps=3000,
                    bs=None,
                    sigma_affine=0.,
                    l1_weight=0,
                    use_lr_schedule=False,
                    vflip=False):
    """
    One stage of the PTI inversion.
    :param model: the UVASystem model
    :param image_batch: the image batch[num_images, 3, h, w]
    :param video_id: the video_id for the image batch, None if not used
    :param only_embedding: whether to optimize only the embedding
    :param preserve_geometry_weight: the weight for the geometry preservation loss
    :param lr: the learning rate
    :param steps: the number of steps
    :param bs: the batch size to use in case all images do not fit in the GPU, None if all images used
    :param sigma_affine: the sigma for the random affine transformation
    :param use_lr_schedule: whether to reduce the learning rate
    :param vflip: whether to flip the image vertically
    """

    if video_id is None:
        video_id = torch.zeros(image_batch.shape[0], device=image_batch.device).long()

    if only_embedding:
        optimizer = torch.optim.Adam(model.canonical_generator.embedding.parameters(), lr=lr, betas=(0.5, 0.99))
    else:
        optimizer = torch.optim.Adam(model.canonical_generator.parameters(), lr=lr, betas=(0.5, 0.99))

    if use_lr_schedule:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [steps // 4, steps // 2, 3 * steps // 4])

    with torch.no_grad():
        initial_canonical = model.canonical_generator.get_canonical_by_id(video_id)

        if sigma_affine != 0:
            pose = model.estimate_pose(image_batch)
            out = model.render_frame(pose['camera_matrix'], initial_canonical)
            segmentation_batch = out['fg_segmentation'].sum(dim=1, keepdims=True).data.clone()
            del out

        if preserve_geometry_weight == 0:
            del initial_canonical        

    prog_bar = tqdm(range(steps))
    if bs is None or image_batch.shape[0] < bs:
        bs = image_batch.shape[0]

    for i in prog_bar:
        optimizer.zero_grad()
        with torch.no_grad():
            # Select random images from image_batch
            idx = torch.randperm(image_batch.shape[0])[:bs]
            image_current = image_batch[idx]
            video_id_current = video_id[idx]

            # Apply random affine transformation 50% of the time
            if sigma_affine != 0 and random.random() > 0.5:
                transform = Transform(image_current.shape[0], sigma_affine=sigma_affine, vflip=vflip)
                image_current = transform.transform_frame(image_current)
                segmentation_current = segmentation_batch[idx]
                segmentation_current = transform.transform_frame(segmentation_current)
                if segmentation_current.shape[2] == image_current.shape[2]:
                    segmentation_current = segmentation_current
                else:
                    segmentation_current = F.interpolate(segmentation_current, size=image_current.shape[2:])

            else:
                segmentation_current = 1

        with torch.no_grad():
            pose = model.estimate_pose(image_current)

        emb_noise_std = 0.5 * max(0, (1 - i / (steps / 2)))**2 if only_embedding else None
        emb = model.canonical_generator.embedding(video_id_current)

        if emb_noise_std is not None:
            noise = torch.normal(mean=torch.zeros_like(emb)) * emb_noise_std
        else:
            noise = 0

        canonical = model.canonical_generator.get_canonical_with_emb(emb + noise)
        out = model.render_frame(pose['camera_matrix'], canonical)
        perceptual_loss = model.perceptual_pyramide(segmentation_current * image_current,
                                                    segmentation_current * out['radiance'])
        loss = perceptual_loss
 
        if l1_weight != 0:
            l1_loss = l1_weight * torch.abs(segmentation_current * image_current - segmentation_current * out['radiance'])
            loss = loss + l1_loss.mean()

        if preserve_geometry_weight:
            geometry_channels = list(range(model.n_parts)) + [-1]
            geometry_loss = torch.abs(canonical[:, geometry_channels] - initial_canonical[:, geometry_channels]).mean()
            loss = geometry_loss * preserve_geometry_weight + loss

        prog_bar.set_description("Loss:" + str(float(loss.data.cpu().numpy())))
        loss.backward()
        optimizer.step()
        if use_lr_schedule:
            scheduler.step()


def read_video(name, image_shape):
    """
    Read video which can be:
      - a folder with frames
      - '.mp4' and'.gif' files
    :param name: the path to the video or folder with frames
    :param image_shape: the resize shape
    :return: the np array of the video [num_frames, h, w, 3]
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video = [io.imread(os.path.join(name, frames[idx])) for idx in range(num_frames)]
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = io.mimread(name)
    else:
        raise Exception("Unknown file extensions  %s" % name)
    video = [resize(frame, image_shape) for frame in video]

    return video


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--tb_log_path", default='tb_logs', help="path to for tensorboard logs")
    parser.add_argument("--checkpoint", required=True, help="path to checkpoint to restore")
    parser.add_argument('--source_images', nargs='+', required=True, help='path to source images')
    parser.add_argument("--driving_video", default=None, help="optional driving video for transfer testing")
    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    state_dict = torch.load(opt.checkpoint)['state_dict']

    model = UVASystem(resolution=config['video_resolution'],
                      num_channels=config['num_channels'],
                      max_videos=state_dict['canonical_generator.embedding.embedding.weight'].shape[0],
                      model_params=config['model_params'],
                      optimizer_params=config['optimizer_params'],
                      loss_params=config['loss_params'])
    model.load_state_dict(state_dict)

    if opt.driving_video is not None:
        driving_video = read_video(opt.driving_video, config['video_resolution'])
        driving_video = torch.Tensor(driving_video).permute(0, 3, 1, 2)
    else:
        driving_video = None
    
    source_images = [resize(io.imread(source_image), config['video_resolution']) for source_image in opt.source_images]
    source_images = torch.Tensor(source_images).permute(0, 3, 1, 2).cuda()

    model.cuda()
    init_model(model)
    inversion_params = config['inversion_params']

    inversion_stage(model,
                    source_images,
                    video_id=None,
                    only_embedding=True,
                    preserve_geometry_weight=0,
                    lr=inversion_params['lr_first'],
                    steps=inversion_params['steps_first'],
                    bs=inversion_params['bs'],
                    l1_weight=inversion_params['l1_first'],
                    sigma_affine=inversion_params['sigma_first'],
                    use_lr_schedule=inversion_params['use_lr_schedule_first'],
                    vflip=inversion_params['vflip_first'])
    model.eval()
    tb = SummaryWriter(opt.tb_log_path)
    visualizer = VisualizationCallback(dataset=None)
    video_dict = {'frame': source_images[:1], 'video_id': torch.zeros(source_images.shape[0]).long().cuda()}
    visualization = visualizer.make_reconstruction(model=model, video=video_dict)
    tb.add_images('first_stage_reconstruction', visualization, 0)
    visualization = visualizer.make_novel_view(model=model, video=video_dict)
    tb.add_video('first_stage_camera', visualization.unsqueeze(0), 0)

    model.train()
    inversion_stage(model,
                    source_images,
                    video_id=None,
                    only_embedding=False,
                    preserve_geometry_weight=inversion_params['preserve_geometry_weight'],
                    lr=inversion_params['lr_second'],
                    steps=inversion_params['steps_second'],
                    bs=inversion_params['bs'],
                    l1_weight=inversion_params['l1_second'],
                    sigma_affine=inversion_params['sigma_second'],
                    use_lr_schedule=inversion_params['use_lr_schedule_second'],
                    vflip=inversion_params['vflip_second'])
    model.eval()
    visualization = visualizer.make_reconstruction(model=model, video=video_dict)
    tb.add_images('second_stage_reconstruction', visualization, 0)
    visualization = visualizer.make_novel_view(model=model, video=video_dict)
    tb.add_video('second_stage_camera', visualization.unsqueeze(0), 0)
    
    if driving_video is not None:
        visualization = visualizer.make_transfer(model=model,
                                                 source_video=video_dict,
                                                 driving_video={'frame': driving_video.cuda()})
        tb.add_video('second_stage_transfer', visualization.unsqueeze(0), 0)
        tb.flush()
