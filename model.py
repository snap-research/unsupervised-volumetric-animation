"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import math

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR

from modules.canonical_generator import CanonicalGenerator
from modules.mraa_region_predictor import MRAARegionPredictor
from modules.perceptual_loss import PerceptualPyramideLoss
from modules.pose_estimation import PnPPose
from modules.util import make_coordinate_grid, AffineTransform
from pytorch3d.transforms import axis_angle_to_matrix


class UVASystem(pl.LightningModule):
    def __init__(self, max_videos, num_channels, resolution,
                 model_params=None, optimizer_params=None, loss_params=None):
        """
        :param max_videos: number of videos in the dataset, this number of embeddings will be created
        :param num_channels: number of channels in each image
        :param resolution: resolution used for rendering, same as dataset resolution
        :param model_params: parameters of the architecture
        :param optimizer_params: parameters of the optimization
        :param loss_params: parameters of the losses
        """
        super().__init__()

        if loss_params is None:
            loss_params = {}
        if optimizer_params is None:
            optimizer_params = {}
        if model_params is None:
            model_params = {}

        self.n_parts = model_params['n_parts']
        self.points_per_part = model_params['points_per_part']
        self.max_videos = max_videos

        self.resolution = resolution
        self.optimizer_params = optimizer_params

        self.loss_params = loss_params

        self.near = model_params['near']
        self.far = model_params['far']
        self.fov = model_params['fov']

        self.rendering_features = num_channels

        self.grad_clip = optimizer_params['grad_clip']
        self.random_points3d_init = model_params['random_points3d_init']

        self.canonical_generator = CanonicalGenerator(n_parts=self.n_parts,
                                                      max_videos=max_videos,
                                                      rendering_features=self.rendering_features,
                                                      **model_params['canonical_generator'])
        self.pnp_pose = PnPPose(n_parts=self.n_parts,
                                points_per_part=self.points_per_part,
                                **model_params['pnp_pose'])

        if loss_params['mraa_mask']['weight']:
            self.mraa_predictor = MRAARegionPredictor(**loss_params['mraa_mask']['region_predictor_params'])
            self.mraa_predictor.load_state_dict(
                torch.load(loss_params['mraa_mask']['checkpoint_path'], map_location=next(self.parameters()).device)[
                    'region_predictor'])
        else:
            self.mraa_predictor = None

        if 'perceptual_pyramide' in loss_params and sum(loss_params['perceptual_pyramide']['weights']) != 0:
            self.perceptual_pyramide = PerceptualPyramideLoss(loss_params['perceptual_pyramide']['scales'],
                                                              loss_params['perceptual_pyramide']['weights'],
                                                              num_channels)

        else:
            self.perceptual_pyramide = None

        self.points3d = None
        self.camera_intrinsics = None
        self.bounds = None
        self.automatic_optimization = False
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Perform initialization of the 3d points, camera intrinsics and scene bounds.
        """

        focal_x = 2 / (2 * math.tan(self.fov / 2))
        focal_y = 2 / (2 * math.tan(self.fov / 2))

        bounds = np.zeros((2, 3))

        bounds[:, 2] = (self.near, self.far)
        bounds[:, 0] = (-self.far / focal_x, self.far / focal_x)
        bounds[:, 1] = (-self.far / focal_y, self.far / focal_y)

        self.bounds = nn.Parameter(torch.FloatTensor(bounds), requires_grad=False)

        self.camera_intrinsics = np.array([
            [focal_x, 0, 0],
            [0, focal_y, 0],
            [0, 0, 1]
        ])
        self.camera_intrinsics = nn.Parameter(torch.FloatTensor(self.camera_intrinsics), requires_grad=False)

        if not self.random_points3d_init:
            points_per_side = int(round(self.points_per_part ** (1 / 3)))
            assert points_per_side ** 3 == self.points_per_part
            points = torch.linspace(0, 1, steps=points_per_side + 2)[1:-1]
            grid = torch.meshgrid(points, points, points)
            grid = torch.stack(grid, dim=-1)
            grid = torch.logit(grid)
            grid = grid.view(1, 1, self.points_per_part, 3).repeat(1, self.n_parts, 1, 1)
            self.points3d = nn.Parameter(grid, requires_grad=True)
        else:
            anchors = torch.zeros((1, self.n_parts, self.points_per_part, 3)).uniform_(-2, 2)
            self.points3d = nn.Parameter(anchors, requires_grad=True)

    def get_intrinsics(self):
        """
        :return: camera intrinsics[1, 3, 3]
        """
        return self.camera_intrinsics.unsqueeze(0)

    def get_bounds(self):
        """
        :return: scene bounds[2, 3]
        """
        return self.bounds

    def get_points3d(self):
        """
        :return: canonical 3d points[n_parts, points_per_part, 3]
        """
        bounds = self.get_bounds()
        distance = (bounds[1] - bounds[0]).expand(self.points3d.shape)
        start = bounds[0].expand(self.points3d.shape)
        points3d = torch.sigmoid(self.points3d) * distance + start
        return points3d

    def get_bg_camera(self, bs):
        """
        Background camera is always identity

        :param bs: batch size
        :return: bg_camera[bs, 1, 3, 4]
        """
        return torch.eye(3, 4, device=self.device).view(1, 1, 3, 4).view(1, 1, 3, 4).repeat(bs, 1, 1, 1)

    def estimate_pose(self, frame):
        """
        Return camera pose of
        """
        return self.pnp_pose(frame, self.get_points3d(), self.get_intrinsics())

    def render_frame(self, camera_matrix, canonical, bg_camera_matrix=None, compute_normals=False):
        """
        Renders the entire frame, given camera_matrix and canonical.

        :param camera_matrix: the pose of each part[bs, n_parts, 3, 4].
        :param canonical: the canonical volume[bs, n_parts + 3 + 1, voxel_h, voxel_w, voxel_d]
        :param bg_camera_matrix: background camera[bs, 1, 3, 4], None for identity
        :param compute_normals: normal computation incur additional overhead, this option allow to skip it

        :return: output dict with:
             - fg_segmentations [bs, n_parts, h, w]
             - bg_segmentations [bs, 1, h, w]
             - radiance [bs, rendering_features, h, w]
             - depth [bs, 1, h, w] - at test time depth is normalized to [0, 1]
             - occupancy [bs, 1, h, w]
             - fg_density [bs, n_parts, h, w, point_per_ray] - this is not returned at test time
             - bg_density [bs, 1, h, w, points_per_ray] - this is not returned at test time
             - point_lbs [bs, n_parts, h, w, points_per_ray] - this is not returned at test time
             - [optionally] normals [bs, 1, h, w]
        """
        ray_grid = make_coordinate_grid(self.resolution, device=self.device)
        bs = canonical.shape[0]

        if bg_camera_matrix is None:
            bg_camera_matrix = self.get_bg_camera(bs)

        if self.training and (self.trainer is not None):
            current_training_step = self.trainer.global_step
        else:
            current_training_step = -1

        output = self.canonical_generator.render(camera_matrix=camera_matrix,
                                                 bg_camera_matrix=bg_camera_matrix,
                                                 canonical=canonical,
                                                 camera_intrinsics=self.get_intrinsics(),
                                                 ray_grid=ray_grid,
                                                 scene_bounds=self.get_bounds(),
                                                 compute_normals=compute_normals,
                                                 current_training_step=current_training_step)

        if self.training:
            depth_range = output['depth'].min().detach(), output['depth'].max().detach()
            self.log_dict({'stats/depth_min': depth_range[0], 'stats/depth_max': depth_range[1]}, rank_zero_only=True)

            occupancy_range = output['occupancy'].min().detach(), output['occupancy'].max().detach()
            self.log_dict({'stats/occupancy_min': occupancy_range[0], 'stats/occupancy_max': occupancy_range[1]},
                          rank_zero_only=True)

            part_density = output['part_density'].mean(dim=(0, 2, 3, 4)).detach()
            self.log_dict({('stats/lbs' + str(i).zfill(2)): part_density[i] for i in range(part_density.shape[0])},
                          rank_zero_only=True)

            fg_density = output['fg_density'].mean()
            self.log('stats/fg_density', fg_density.detach(), rank_zero_only=True)
        else:
            # Normalize depth to [0, 1]
            bounds = self.get_bounds()
            output['depth'] = (output['depth'] - bounds[0, -1]) / (bounds[1, -1] - bounds[0, -1])
            # Remove unnecessary outputs
            del output['fg_density']
            del output['bg_density']
            del output['part_density']

        return output

    def make_novel_view_matrix(self, angles, translation=0, center=None):
        """
        Generate a novel view matrix given axis angles translation and center.
        The matrix will rotate camera around the center with provided angles and then translate.

        :param angles: axis angles [bs, 3] for the camera rotation around the center
        :param translation: the relative translation [bs, 3] of the camera
        :param center: the center[bs, 3] around which to rotate the camera,
                       if not specified use center computed from scene bounds
        :return: novel view matrix [bs, 3, 4]
        """
        bounds = self.get_bounds()

        if center is None:
            center = (bounds[:1] + bounds[1:]) / 2

        centering_transform = torch.eye(4, device=bounds.device).unsqueeze(0)
        centering_transform[:, :3, 3] = center
        un_centering_transform = torch.eye(4, device=bounds.device).unsqueeze(0)
        un_centering_transform[:, :3, 3] = -center

        novel_view_matrix = torch.eye(4, device=angles.device).unsqueeze(0)
        rotation_matrix = axis_angle_to_matrix(angles)
        novel_view_matrix[:, :3, :3] = rotation_matrix
        novel_view_matrix[:, :3, 3] = translation

        novel_view_matrix = centering_transform @ novel_view_matrix @ un_centering_transform
        return novel_view_matrix

    def generate_novel_view(self, camera_matrix, canonical, novel_view_matrix, transform_bg=True,
                            compute_normals=False):
        """
        Render novel view.

        :param camera_matrix: the pose of each part[bs, n_parts, h, w]
        :param canonical: the canonical volume[bs, n_parts + 3 + 1, voxel_h, voxel_w, voxel_d]
        :param novel_view_matrix: the matrix for a new camera pose[bs, 3, 4]
        :param transform_bg: True if background need to be rendered under novel view
        :param compute_normals: True if normals is needed

        :return: output dict (same as self.render_frame)
        """
        camera_matrix = camera_matrix @ novel_view_matrix.unsqueeze(1)

        if transform_bg:
            bg_camera_matrix = self.get_bg_camera(camera_matrix.shape[0]) @ novel_view_matrix.unsqueeze(1)
        else:
            bg_camera_matrix = None

        generated = self.render_frame(camera_matrix=camera_matrix,
                                      canonical=canonical,
                                      bg_camera_matrix=bg_camera_matrix,
                                      compute_normals=compute_normals)
        return generated

    def training_step(self, x):
        optimizer = self.optimizers()
        optimizer.zero_grad()

        pose = self.estimate_pose(x['frame'])
        canonical = self.canonical_generator.get_canonical_by_id(x['video_id'])
        rendering = self.render_frame(pose['camera_matrix'], canonical)

        loss = 0
        if 'projection' in self.loss_params and self.loss_params['projection']['weight']:
            points2d = pose['points2d']
            points_proj = pose['points_proj']
            value = torch.abs(points2d - points_proj)
            projection = self.loss_params['projection']['weight'] * value.mean()
            self.log('loss/projection', projection.detach().mean(), rank_zero_only=True)
            loss = loss + projection

        if 'equivariance' in self.loss_params and self.loss_params['equivariance']['weight']:
            bs = x['frame'].shape[0]
            transform = AffineTransform(bs, self.loss_params['equivariance']['sigma_affine'])
            transformed_frame = transform.transform_frame(x['frame'])
            points2d = self.pnp_pose.get_points_2d(transformed_frame)['points2d']
            value = torch.abs(pose['points2d'].view(bs, -1, 2) -
                              transform.warp_coordinates(points2d.view(bs, -1, 2))).mean()
            equivariance = self.loss_params['equivariance']['weight'] * value
            self.log('loss/equivariance', equivariance.detach().mean(), rank_zero_only=True)
            loss = loss + equivariance

        if self.perceptual_pyramide is not None:
            perceptual = self.perceptual_pyramide(x['frame'], rendering['radiance'])
            self.log('loss/perceptual_pyramide', perceptual.data.cpu(), rank_zero_only=True)
            loss = loss + perceptual

        if 'pull' in self.loss_params and self.loss_params['pull']['weight']:
            segmentation_dist = rendering['part_density']
            segmentation_dist = segmentation_dist.mean((0, 2, 3, 4)).detach()
            weight = torch.relu(self.loss_params['pull']['threshold'] - segmentation_dist)

            largest_part = torch.argmax(segmentation_dist)
            largest_weight = torch.max(segmentation_dist)
            # When weight for all parts is too small pull torwards background
            if largest_weight < self.loss_params['pull']['threshold']:
                largest_part_pose = self.get_bg_camera(bs=1)
            else:
                largest_part_pose = pose['camera_matrix'][:, largest_part:(largest_part + 1)].detach()
            value = weight * torch.abs(pose['camera_matrix'] - largest_part_pose).mean(dim=(0, 2, 3))
            pull = self.loss_params['pull']['weight'] * value.mean()
            self.log('loss/pull', pull.detach().mean(), rank_zero_only=True)
            loss = loss + pull

        if 'mraa_mask' in self.loss_params and self.loss_params['mraa_mask']['weight'] != 0:
            with torch.no_grad():
                mask, class_weights = self.mraa_predictor(x['frame'])

            if not self.loss_params['mraa_mask']['rebalanced']:
                class_weights = None

            hard_mask = F.interpolate(mask, size=self.resolution, mode='bilinear', align_corners=True)
            predicted_mask = torch.cat([rendering['fg_segmentation'], rendering['bg_segmentation']], dim=1)
            predicted_mask = predicted_mask / torch.clamp(predicted_mask.sum(dim=1, keepdim=True), min=1e-5)

            value = F.cross_entropy(torch.clamp(predicted_mask, 1e-5, 1 - 1e-5), hard_mask, weight=class_weights)

            mraa_mask = value * self.loss_params['mraa_mask']['weight']
            self.log('loss/mraa_mask', mraa_mask.data.cpu(), rank_zero_only=True)

            decay = self.loss_params['mraa_mask']['decay'] ** self.trainer.current_epoch
            self.log('stats/mraa_mask_decay', decay, rank_zero_only=True)
            loss = loss + mraa_mask * decay

        self.log('loss/total_loss', loss.data.cpu(), prog_bar=True, rank_zero_only=True)

        self.manual_backward(loss)
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.grad_clip)
        optimizer.step()

    def training_epoch_end(self, outputs):
        """
        Manual handling of the schedules.
        """
        schedulers = self.lr_schedulers()
        try:
            for sch in schedulers:
                sch.step()
        except TypeError:
            schedulers.step()

    def load_state_dict(self, state_dict):
        """
        Custom state-dict loading that support loading model
        trained with other resolution and model from the first stage.
        """
        with torch.no_grad():
            self_state = self.state_dict()
            for name, param in state_dict.items():
                # Do not load mraa_predictor and vgg
                if name.startswith('mraa_predictor') or name.startswith('perceptual_pyramide'):
                    continue
                # Do not load anti-aliasing parameters
                if name.startswith('pnp_pose.down'):
                    continue

                # Expanding one-part to all parts
                if name.startswith('canonical_generator.decoder.final_conv') and self_state[name].shape != param.shape:
                    weight = param.data
                    repeats = [1] * len(weight.shape)
                    repeats[0] = self.n_parts
                    param = torch.cat([weight[:1].repeat(*repeats), weight[(-self.rendering_features - 1):]], dim=0)

                if name.startswith("pnp_pose.points") and self_state[name].shape != param.shape:
                    weight = param.data
                    repeats = [1] * len(weight.shape)
                    repeats[0] = self.n_parts
                    param = weight.repeat(*repeats)

                if name.startswith("points3d") and self_state[name].shape != param.shape:
                    weight = param.data
                    repeats = [1] * len(weight.shape)
                    repeats[1] = self.n_parts
                    param = weight.repeat(*repeats)

                # If number of embedding did not match copy only the first
                if name.startswith("canonical_generator.embedding.embedding") and self_state[name].shape != param.shape:
                    size = min(self_state[name].shape[0], param.shape[0])
                    self_state[name][:size] = param[:size]
                else:
                    if isinstance(param, nn.Parameter):
                        param = param.data
                    self_state[name].copy_(param)

    def load_legacy_state_dict(self, state_dict):
        """
        Load legacy model with old naming convention.
        """
        with torch.no_grad():
            self_state = self.state_dict()
            for name, param in state_dict.items():
                # Do not load mraa_predictor and vgg
                if name.startswith('mraa_predictor') or name.startswith('perceptual_pyramide'):
                    continue

                # Do not load anti-aliasing parameters
                if name.startswith('movement_predictor'):
                    name = name.replace('movement_predictor', 'pnp_pose')
                    name = name.replace('encoder.', '')
                    name = name.replace('decoder.', '')

                if name.startswith('nerf'):
                    name = name.replace('nerf', 'canonical_generator')
                    name = name.replace('lattent_decoder', 'decoder')
                    name = name.replace('video_embedding_layer', 'embedding')

                not_load = name.startswith('log_K') or \
                           name.startswith('pnp_pose.average_camera_pose') or \
                           name.startswith('depth_range') or \
                           name.startswith('movement_predictor.average_camera_pose') or \
                           name.startswith('canonical_generator.oob')

                if not_load:
                    continue

                if name.startswith('anchors'):
                    name = name.replace('anchors', 'points3d')

                if name.startswith('canonical_generator.decoder.final_conv'):
                    if param[3:-2].shape[0] == 0:
                        param = torch.cat([param[3:-1], param[:3], param[-2:-1]], axis=0)
                    else:
                        param = torch.cat([param[3:-2], param[:3], param[-2:-1]], axis=0)

                if isinstance(param, nn.Parameter):
                    param = param.data

                if self_state[name].shape != param.shape:
                    message = "The shapes ({shape1, shape2}) of the parameter {name} did not match"
                    message = message.format(name=name, shape1=param.shape, shape2=self_state[name].shape)
                    raise IndexError(message)

                self_state[name].copy_(param)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.optimizer_params['lr'],
                                     betas=self.optimizer_params['betas'])

        scheduler = MultiStepLR(optimizer,
                                self.optimizer_params['epoch_milestones'],
                                gamma=self.optimizer_params['gamma'])
        return [optimizer], [scheduler]
