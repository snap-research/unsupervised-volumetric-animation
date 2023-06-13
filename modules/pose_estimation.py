"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import warnings

import torch
import torch.nn.functional as F
from torch import nn

from modules.blocks import Hourglass, AntiAliasInterpolation2d
from modules.util import make_coordinate_grid
from modules.util3d import project3d, to_homogeneous
from pytorch3d.ops.perspective_n_points import efficient_pnp


def softargmax(heatmap):
    shape = heatmap.shape
    grid = make_coordinate_grid(shape[2:], device=heatmap.device)
    heatmap = heatmap.unsqueeze(-1)
    grid = grid.unsqueeze(1)
    mean = (heatmap * grid).sum(dim=(2, 3))
    return mean


class PnPPose(nn.Module):
    """
    Camera pose prediction using PnP. (see Sec. 3.2)
    """

    def __init__(self,
                 n_parts,
                 points_per_part=4,
                 weighted=False,
                 block_expansion=32,
                 num_channels=3,
                 max_features=1024,
                 num_blocks=5,
                 temperature=0.1,
                 scale_factor=0.25):
        """
        PnP-related parameters:
        :param n_parts: number of parts in the object
        :param points_per_part: number of points in each part
        :param weighted: use weighted version of pnp

        Network-related parameters:
        :param block_expansion: multiplicator of the conv layers features size
        :param num_channels: number of channels in the image
        :param max_features: maximum number of features in one conv layer
        :param num_blocks: number of blocks in the network
        :param temperature: keypoint extraction soft-argmax temperature
        :scale_factor: network operates on images of smaller resolution, how much to scale
        """
        super().__init__()

        self.predictor = Hourglass(in_features=num_channels,
                                   block_expansion=block_expansion,
                                   max_features=max_features,
                                   num_blocks=num_blocks)

        self.points = nn.Conv2d(in_channels=self.predictor.out_filters,
                                out_channels=n_parts * points_per_part,
                                kernel_size=(7, 7),
                                padding=3)
        if weighted:
            self.point_weights = nn.Conv2d(in_channels=self.predictor.out_filters,
                                           out_channels=n_parts * points_per_part,
                                           kernel_size=(7, 7),
                                           padding=3)

        self.temperature = temperature
        self.scale_factor = scale_factor
        self.points_per_part = points_per_part
        self.n_parts = n_parts
        self.weighted = weighted

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def get_points_2d(self, image):
        """
        Extract 2d point from image.

        :return: A dict with 'point2d' [bs, n_parts, point_per_part, 2] and
                 optionally point weights [bs, n_parts, point_per_part, 1]
        """
        if self.scale_factor != 1:
            image = self.down(image)

        feature_map = self.predictor(image)
        prediction = self.points(feature_map)

        final_shape = prediction.shape
        raw = prediction.view(final_shape[0], final_shape[1], -1)
        normalized = F.softmax(raw / self.temperature, dim=2)

        out = {}
        if self.weighted:
            weights = self.point_weights(feature_map).view(final_shape[0], final_shape[1], -1)
            weights = (normalized * torch.sigmoid(weights)).sum(dim=2)
            out['weights'] = weights

        normalized = normalized.view(*final_shape)
        points2d = softargmax(normalized)
        out['points2d'] = points2d.view(image.shape[0], self.n_parts, self.points_per_part, 2)

        return out

    def forward(self, image, points3d, camera_intrinsics):
        """
        Extract pose for each part from image

        :param image: the image [bs, 3, h, w] from which to extract the pose
        :param points3d: the 3d points [1, n_parts, points_per_part, 3] from canonical space
        :param camera_intrinsics: the camera_intrinsics matrix [bs, 3, 3]
        """
        bs = image.shape[0]
        out = self.get_points_2d(image)
        points2d = out['points2d'].view(bs * self.n_parts, self.points_per_part, 2)

        if 'weights' in out:
            weights = out['weights'].view(bs * self.n_parts, self.points_per_part)
            weights = weights / torch.max(weights, dim=-1, keepdim=True)[0]
        else:
            weights = None

        points3d = points3d.repeat(bs, 1, 1, 1)
        out['points3d'] = points3d

        points3d = points3d.view(bs * self.n_parts, self.points_per_part, 3)

        camera_intrinsics_inv = torch.inverse(camera_intrinsics)
        repeat_shape = points3d.shape[0] // camera_intrinsics_inv.shape[0]
        camera_intrinsics_inv = camera_intrinsics_inv.repeat(repeat_shape, 1, 1).unsqueeze(1)
        points2d = torch.matmul(camera_intrinsics_inv, to_homogeneous(points2d).unsqueeze(-1)).squeeze(-1)[..., :2]

        if self.training and points2d.requires_grad:
            points2d.register_hook(lambda x: torch.nan_to_num(x))
            points3d.register_hook(lambda x: torch.nan_to_num(x))
            if weights is not None:
                weights.register_hook(lambda x: torch.nan_to_num(x))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pnp = efficient_pnp(points3d, points2d, weights)

        # pnp return transposed matrix R and world-to-camera [R,T]
        # so first we need to transpose R to get world-to-camera, then inverse it to get camera-to-world
        # R is unitary so inverse and transpose is the same
        rotation = pnp.R
        # we need to inverse T to get camera-to-world
        translation = -torch.matmul(rotation, pnp.T.unsqueeze(-1))
        camera_matrix = torch.cat([rotation, translation], dim=-1)

        out['camera_matrix'] = camera_matrix.view(bs, self.n_parts, 3, 4)
        out['points_proj'] = project3d(out['points3d'], out['camera_matrix'], camera_intrinsics)

        if self.training and hasattr(self, 'log'):
            self.log('stats/error_2d', pnp.err_2d.mean().detach(), rank_zero_only=True)
            self.log('stats/spread_2d',
                     torch.abs(points2d - points2d.mean(axis=1, keepdims=True)).mean(),
                     rank_zero_only=True)
            self.log('stats/spread_3d',
                     torch.abs(points3d - points3d.mean(axis=1, keepdims=True)).mean(),
                     rank_zero_only=True)
            if weights is not None:
                self.log('stats/min_weight', out['weights'].min(), rank_zero_only=True)
                self.log('stats/max_weight', out['weights'].max(), rank_zero_only=True)

        return out
