"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from skimage.draw import disk, rectangle, rectangle_perimeter

from modules.util import draw_colored_heatmap, make_coordinate_grid


class VisualizationCallback(Callback):

    def __init__(self,
                 dataset,
                 colormap='gist_rainbow',
                 bg_color=(1, 1, 1),
                 kp_size=5,
                 video_frequency=10,
                 camera_angle_range=(-0.5, 0.5),
                 camera_angle_steps=64):
        """
        Callback used for visualization, relies on tensorboard.

        :param dataset: dataset from which visualization videos will be used
        :param colormap: colormap that is used for segmentation maps visualization and points visualization
        :param bg_color: bg_color for segmentation visualization
        :param video_frequency: how often to save the video output (default is every 10 epochs)

        Camera visualization:
        :param camera_angle_range: range for the change in yaw for camera visualization
        :param camera_angle_steps: number of steps for camera visualization
        """
        super().__init__()
        self.video_frequency = video_frequency
        self.colormap = plt.get_cmap(colormap)
        self.bg_color = np.array(bg_color)
        self.dataset = dataset
        self.kp_size = kp_size

        self.camera_angle_range = camera_angle_range
        self.camera_angle_steps = camera_angle_steps

    def on_train_epoch_start(self, trainer, model):
        """
        We save visualization before each epoch.
        """

        # We don't want this to run on other gpus than 0
        if model.global_rank != 0:
            return
        model.eval()
        if (trainer.current_epoch + 1) % self.video_frequency == 0:
            self.make_video_visualization(trainer.current_epoch, model)
        self.make_image_visualization(trainer.current_epoch, model)
        model.train()

    def make_frame_with_points(self, pose, frame):
        """
        Create points visualization, for each part it creates a separate image.
        On this image all points corresponding to this part is shown.
        Circles is used for predicted 2d points, and squares is used for 2d projections of 3d points.
        Color of border around the frame correspond to the point color.

        :param pose: pose from pnp_pose
        :param frame: frame [bs, 3, h, w]  on which to render points

        return frame with points [bs, 3, h, w * n_parts]
        """
        image = frame.data.cpu().permute(0, 2, 3, 1).numpy()
        points2d = pose['points2d']
        points_proj = pose['points_proj']
        bs, n_parts, points_per_part, _ = points2d.shape

        spatial_size = torch.tensor(image.shape[1:3][::-1])[None, None, None].type(points2d.type())
        points2d = spatial_size * (points2d + 1) / 2
        points_proj = spatial_size * (points_proj + 1) / 2

        points2d = points2d.view(bs, n_parts, points_per_part, 2).data.cpu().numpy()
        points_proj = points_proj.view(bs, n_parts, points_per_part, 2).data.cpu().numpy()

        images = []
        image_copy = image

        for j in range(n_parts):
            image = image_copy.copy()

            # Draw border of color corresponding to the part
            rr, cc = rectangle_perimeter((-1, -1), (10000, 10000), shape=image.shape[1:3], clip=True)
            image[:, rr, cc] = np.array(self.colormap(j / n_parts))[:3]

            # Draw set of points for each part, point2d is circles while points_proj is squares
            for i in range(bs):
                for kp_ind, (kp2d, kp3d) in enumerate(zip(points2d[i, j], points_proj[i, j])):
                    rr, cc = disk((kp2d[1], kp2d[0]), self.kp_size, shape=image.shape[1:3])
                    image[i, rr, cc] = np.array(self.colormap(kp_ind / points_per_part))[:3]

                for kp_ind, (kp2d, kp3d) in enumerate(zip(points2d[i, j], points_proj[i, j])):
                    half_kp = self.kp_size // 2 + 1

                    rr, cc = rectangle((int(round(kp3d[1] - half_kp)), int(round(kp3d[0] - half_kp))),
                                       extent=(self.kp_size + 2, self.kp_size + 2),
                                       shape=image.shape[1:3])

                    if len(rr) != 0 and len(cc) != 0:
                        image[i, rr, cc] = 1

                    half_kp = self.kp_size // 2
                    rr, cc = rectangle((int(round(kp3d[1] - half_kp)), int(round(kp3d[0] - half_kp))),
                                       extent=(self.kp_size, self.kp_size),
                                       shape=image.shape[1:3])

                    if len(rr) != 0 and len(cc) != 0:
                        image[i, rr, cc] = np.array(self.colormap(kp_ind / points_per_part))[:3]

            images.append(image)
        image = np.concatenate(images, axis=2)
        return torch.tensor(image, device=frame.device).permute(0, 3, 1, 2)

    def stack_output(self, all_outputs):
        """
        Create a large stacked video of all intermediate outputs.

        :param all_outputs: a list  of dicts for each frame, each dict hast the structure:
                                                             'radiance': [bs, 3, h, w]
                                                             'gt_frame': [bs, 3, h, w]
                                                             ...
        :return: image containing all the frames stacked
        """
        out = defaultdict(list)
        for frame in all_outputs:
            for key, value in frame.items():
                out[key].append(value)

        for key, value in out.items():
            out[key] = torch.cat(value, dim=0)

        result = []
        final_shape = out['radiance'].shape

        def add_to_result(image):
            shape = list(image.shape)
            shape[3] = (shape[3] // shape[2]) * final_shape[2]
            shape[2] = final_shape[2]
            image = F.interpolate(image, size=shape[2:], mode='bilinear', align_corners=False)
            if image.shape[1] != final_shape[1]:
                image = image.repeat(1, final_shape[1] // image.shape[1], 1, 1)
            result.append(torch.clamp(image, 0, 1).data.cpu())

        if 'gt_frame' in out:
            add_to_result(out['gt_frame'])

        if 'radiance' in out:
            add_to_result(out['radiance'])

        if 'novel_view_frame' in out:
            add_to_result(out['novel_view_frame'])

        if 'fg_segmentation' in out:
            fg_segmentation = out['fg_segmentation']
            bg_segmentation = out['bg_segmentation']
            segmentation = draw_colored_heatmap(fg_segmentation, bg_segmentation, self.colormap, self.bg_color)
            add_to_result(segmentation)

        if 'gt_segmentation' in out:
            fg_segmentations = out['gt_segmentation'][:, :-1]
            bg_segmentations = out['gt_segmentation'][:, -1:]
            gt_segmentation = draw_colored_heatmap(fg_segmentations, bg_segmentations, self.colormap, self.bg_color)
            add_to_result(gt_segmentation)

        if 'depth' in out:
            add_to_result(out['depth'])

        if 'occupancy' in out:
            add_to_result(out['occupancy'])

        if 'normals' in out:
            normals = (out['normals'] + 1) / 2
            add_to_result(normals)

        if 'image_with_points' in out:
            add_to_result(out['image_with_points'])

        result = torch.cat(result, axis=3)
        return result

    def make_video_visualization(self, epoch, model, tb=None):
        """
        Create a video visualization by passing over all video frames.

        :param epoch: current epoch
        :param model: UVASystem model
        :param tb: tensorboard logger, if None will be taken from model
        """
        videos = self.dataset.get_visualization_videos()
        if tb is None:
            tb = model.logger.experiment

        videos = [{key: value.cuda() for key, value in video.items()} for video in videos]

        for i, video in enumerate(videos):
            tb.add_video('reconstruction/videos' + str(i), self.make_reconstruction(model, video).unsqueeze(0), epoch)
            tb.add_video('random/videos' + str(i), self.make_random(model, video).unsqueeze(0), epoch)
            tb.add_video('camera/videos' + str(i), self.make_novel_view(model, video).unsqueeze(0), epoch)

        for i, (source, driving) in enumerate(zip(videos, videos[::-1])):
            tb.add_video('transfer/' + str(i), self.make_transfer(model, source, driving).unsqueeze(0), epoch)

    def make_image_visualization(self, epoch, model, tb=None):
        """
        Create an image visualization by considering only the first frame.

        :param epoch: current epoch
        :param model: UVASystem model
        :param tb: tensorboard logger, if None will be taken from model
        """
        videos = self.dataset.get_visualization_videos()
        if tb is None:
            tb = model.logger.experiment

        # Take only the first frame
        videos = [{key: value[:1].cuda() for key, value in video.items()} for video in videos]

        for i, video in enumerate(videos):
            tb.add_images('reconstruction/images' + str(i), self.make_reconstruction(model, video), epoch)
            tb.add_images('canonical/' + str(i), self.make_canonical(model, video), epoch)
            tb.add_images('random/' + str(i), self.make_random(model, video), epoch)

        self.draw_rays(epoch, model, videos[0], tb=tb)

    def make_reconstruction(self, model, video):
        """
        Create a reconstruction visualization for a single video

        :param model: UVASystem model
        :param video: video to be reconstructed
        :return: large stacked video of everything [n_frames, 3, h, w x num_visualization]
        """
        # Use first frame id to extract canonical
        canonical = model.canonical_generator.get_canonical_by_id(video['video_id'][:1])

        out_list = []
        for i in range(video['frame'].shape[0]):
            pose = model.estimate_pose(video['frame'][i:(i + 1)])
            out = model.render_frame(pose['camera_matrix'], canonical, compute_normals=True)

            out['gt_frame'] = video['frame'][i:(i + 1)]
            out['image_with_points'] = self.make_frame_with_points(pose, video['frame'][i:(i + 1)])

            angles = torch.tensor([[0, self.camera_angle_range[0], 0]]).to(video['video_id'].device)
            novel_view_matrix = model.make_novel_view_matrix(angles=angles)
            novel_view = model.generate_novel_view(pose['camera_matrix'], canonical, novel_view_matrix)
            out['novel_view_frame'] = novel_view['radiance']

            if model.mraa_predictor:
                out['gt_segmentation'], _ = model.mraa_predictor(video['frame'][i:(i + 1)])

            out = {key: value.data.cpu() for key, value in out.items()}
            out_list.append(out)
        return self.stack_output(out_list)

    def make_random(self, model, video):
        """
        Create a random identity visualization for a single driving video

        :param model: UVASystem model
        :param video: video that will be used as driving
        :return: large stacked video of everything [n_frames, 3, h, w x num_visualization]
        """

        canonical = model.canonical_generator.get_random_canonical(num_random=1)

        out_list = []
        for i in range(video['frame'].shape[0]):
            pose = model.estimate_pose(video['frame'][i:(i + 1)])
            out = model.render_frame(pose['camera_matrix'], canonical, compute_normals=True)
            out['gt_frame'] = video['frame'][i:(i + 1)]
            del out['occupancy']

            out = {key: value.data.cpu() for key, value in out.items()}
            out_list.append(out)

        return self.stack_output(out_list)

    def make_canonical(self, model, video):
        """
        Create a random canonical visualization for a single 1-frame video

        :param model: UVASystem model
        :param video: 1-frame video that will be used for extracting identity
        :return: large stacked video of everything [1, 3, h, w x num_visualization]
        """
        out_list = []
        canonical = model.canonical_generator.get_canonical_by_id(video['video_id'][:1])

        camera_matrix = model.get_bg_camera(1)
        camera_matrix = camera_matrix.repeat(1, model.n_parts, 1, 1)

        out = model.render_frame(camera_matrix, canonical, compute_normals=True)
        del out['occupancy']
        out = {key: value.data.cpu() for key, value in out.items()}
        out_list.append(out)
        return self.stack_output(out_list)

    def make_transfer(self, model, source_video, driving_video):
        """
        Create a transfer visualization where appearance is taken from the source_video and motion from driving_video.

        :param model: UVASystem model
        :param source_video: video that will be used for extracting identity
        :param driving_video: video that will be used as driving
        :return: large stacked video of everything [num_frames, 3, h, w x num_visualization]
        """
        out_list = []
        canonical = model.canonical_generator.get_canonical_by_id(source_video['video_id'][:1])

        for i in range(driving_video['frame'].shape[0]):
            pose = model.estimate_pose(driving_video['frame'][i:(i + 1)])
            out = model.render_frame(pose['camera_matrix'], canonical, compute_normals=True)
            out['gt_frame'] = driving_video['frame'][i:(i + 1)]
            out = {key: value.data.cpu() for key, value in out.items()}
            out_list.append(out)
        return self.stack_output(out_list)

    def make_novel_view(self, model, video):
        """
        Create novel_view visualization, where the novel_view is changing according to camera_angle_range.

        :param model: UVASystem model
        :param video: video that will be used for extracting identity
        :return: large stacked video of everything [num_frames, 3, h, w x num_visualization]
        """
        canonical = model.canonical_generator.get_canonical_by_id(video['video_id'][:1])
        pose = model.estimate_pose(video['frame'][:1])

        out_list = []
        angles = torch.linspace(self.camera_angle_range[0], self.camera_angle_range[1], steps=self.camera_angle_steps)
        angle_axis = torch.zeros((angles.shape[0], 3))
        angle_axis[:, 1] = angles

        for j in range(angles.shape[0]):
            novel_view_matrix = model.make_novel_view_matrix(angles=angle_axis[j].to(video['video_id'].device))
            out = model.generate_novel_view(pose['camera_matrix'], canonical, novel_view_matrix=novel_view_matrix)
            del out['occupancy']
            out = {key: value.data.cpu() for key, value in out.items()}
            out_list.append(out)

        return self.stack_output(out_list)

    def draw_rays(self, epoch, model, video, tb):
        """
        Create rays visualization for a video.

        :param epoch: current epoch
        :param model: UVASystem model
        :param video: video that will be for pose estimation
        :param tb: tensorboard object
        """
        pose = model.estimate_pose(video['frame'])

        camera_matrix = pose['camera_matrix']
        camera_intrinsics = model.get_intrinsics()

        grid = make_coordinate_grid([2, 2], camera_matrix.device)
        rays_o, rays_d = model.canonical_generator.get_rays(camera_matrix, camera_intrinsics, grid)

        near = model.near
        far = model.far
        rays_o = rays_o[0, :, 0, 0].unsqueeze(1)
        d_border = torch.cat([
            rays_d[0, :, 0, 0].unsqueeze(1), rays_d[0, :, 0, -1].unsqueeze(1), rays_d[0, :, -1, 0].unsqueeze(1),
            rays_d[0, :, -1, -1].unsqueeze(1)
        ],
                             dim=1)

        near = d_border * near + rays_o
        far = d_border * far + rays_o
        vec = far - near

        near = near.data.cpu().numpy()
        vec = vec.data.cpu().numpy()

        bounds = model.get_bounds().data.cpu().numpy()
        camera_matrix = camera_matrix.data[0].cpu().numpy()

        num_regions = near.shape[0]
        colors = [self.colormap(i / num_regions) for i in range(num_regions)]

        def make_fig(elev, azim, draw_camera_origin=False):
            fig = plt.figure()
            ax = plt.axes(projection='3d')

            points = map(np.array, itertools.product([0, 1], repeat=3))

            for p1, p2 in itertools.combinations(points, 2):
                if np.abs(p2 - p1).sum() == 1:
                    ax.plot3D(bounds[[p1[0], p2[0]], 0], bounds[[p1[1], p2[1]], 1], bounds[[p1[2], p2[2]], 2], c='red')

            if draw_camera_origin:
                ax.scatter3D(camera_matrix[:, 0, 3],
                             camera_matrix[:, 1, 3],
                             camera_matrix[:, 2, 3],
                             c=colors,
                             cmap=self.colormap,
                             alpha=0.3)

            for i in range(near.shape[1]):
                ax.quiver(near[:, i, 0],
                          near[:, i, 1],
                          near[:, i, 2],
                          vec[:, i, 0],
                          vec[:, i, 1],
                          vec[:, i, 2],
                          colors=colors,
                          alpha=0.3,
                          arrow_length_ratio=0.005)
            ax.view_init(elev, azim)
            return fig

        tb.add_figure('rays_visualization/xz', make_fig(0, 0), epoch)
        tb.add_figure('rays_visualization/rotated1', make_fig(0, 60), epoch)
        tb.add_figure('rays_visualization/rotated2', make_fig(30, 0), epoch)
        tb.add_figure('rays_visualization/xy', make_fig(90, 0), epoch)

        tb.add_figure('rays_visualization/xz-cam', make_fig(0, 0, True), epoch)
        tb.add_figure('rays_visualization/rotated1-cam', make_fig(0, 60, True), epoch)
        tb.add_figure('rays_visualization/rotated2-cam', make_fig(30, 0, True), epoch)
        tb.add_figure('rays_visualization/xy-cam', make_fig(90, 0, True), epoch)

        plt.close('all')
