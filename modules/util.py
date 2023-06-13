"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import numpy as np
import torch
import torch.nn.functional as F


def normalize_histogram(depth, bins=256):
    """
    Given a depth map with arbitrary range of values, normalize it to [0, 1],
    such that every bin of size 1/bins have equal number of samples that fall into it.
    See https://en.wikipedia.org/wiki/Histogram_equalization.

    :param depth: The depth map [bs, 1, h, w] to normalize.
    :param bins: Number of bins for histogram.
    :return: normalized depth map [bs, 1, h, w]
    """
    bs, _, h, w = depth.shape
    values = depth.view(depth.shape[0], h * w)

    quantile = torch.linspace(0, 1, steps=bins + 1, device=values.device)
    lower_bounds = torch.quantile(values, quantile[:-1], dim=1)
    lower_bounds = torch.transpose(lower_bounds, 1, 0)
    upper_bounds = torch.cat([lower_bounds[:, 1:], torch.max(values, dim=1, keepdim=True)[0]], axis=-1)
    index = torch.searchsorted(upper_bounds, values, right=False)

    lower_bounds = torch.gather(lower_bounds, dim=1, index=index)
    upper_bounds = torch.gather(upper_bounds, dim=1, index=index)

    new_values = (values - lower_bounds) / (upper_bounds - lower_bounds + 1e-5)  # Values from 0 to 1
    new_values = (new_values + index) / bins

    return new_values.view(bs, 1, h, w)


def normalize_01(depth):
    """
    Given depth normalize it to [0, 1] range

    :param depth: The depth map to normalize
    :return: normalized depth map
    """
    low, high = depth.min(), depth.max()
    return (depth - low) / (high - low)


def draw_colored_heatmap(heatmap, bg_weight, colormap, bg_color=(1, 1, 1)):
    """
    Convert multi-chanel heatmap to RGB colored image.

    :param heatmap: The heatmap [bs, c, h, w] map to normalize
    :param bg_weight: The heatmap [bs, 1, h, w] for the background.
    :param colormap: The plt colormap to use for coloring.
    :param bg_color: RGB color of the background.
    :return: RGB image [bs, 3, h, w]
    """
    parts = []
    bg_color = torch.Tensor(np.array(bg_color).reshape((1, 3, 1, 1))).type(heatmap.type())
    num_regions = heatmap.shape[1]
    for i in range(num_regions):
        color = torch.Tensor(np.array(colormap(i / num_regions))[:3]).type(heatmap.type())
        color = color.reshape((1, 3, 1, 1))
        part = heatmap[:, i:(i + 1)]
        color_part = part * color
        parts.append(color_part)

    result = sum(parts) * (1 - bg_weight) + bg_weight * bg_color
    return result


def make_coordinate_grid(spatial_size, device):
    """
    Create a meshgrid [-1,1] x [-1,1].

    :param spatial_size: size of the resulting meshgrid.
    :param device: device where to put the meshgrid.
    :return: coordinate grid [bs, spatial_size[0], spatial_size[1], 2]
    """
    theta = torch.eye(2, 3, device=device).unsqueeze(0)
    return torch.nn.functional.affine_grid(theta, size=[1, 1, spatial_size[0], spatial_size[1]], align_corners=True)


class AffineTransform:
    """
    Random affine transformation for equivariance constraints.
    """

    def __init__(self, bs, sigma_affine=0.05):
        noise = torch.normal(mean=0, std=sigma_affine * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)
        self.bs = bs

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], device=frame.device)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection", align_corners=True)

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)
        return transformed
