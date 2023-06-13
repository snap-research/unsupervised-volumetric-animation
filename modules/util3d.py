"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch


def to_homogeneous(coordinates):
    """
    Convert regular coordinate to homogeneous

    :param coordinates: Regular coordinates [..., 2]
    :return: Homogeneous coordinates [..., 3]
    """
    ones_shape = list(coordinates.shape)
    ones_shape[-1] = 1
    ones = torch.ones(ones_shape).type(coordinates.type())

    return torch.cat([coordinates, ones], dim=-1)


def from_homogeneous(coordinates):
    """
    Convert homogeneous coordinate to regular

    :param coordinates: Homogeneous coordinates [..., 3]
    :return: Regular coordinates [..., 3]
    """
    return coordinates[..., :2] / coordinates[..., 2:3]


def project3d(points3d, camera_matrix, camera_intrinsics):
    """
    Project 3d points into 2d image plane, using the camera_matrix and camera_intrinsics.

    :param points3d: [..., num_points, 3]
    :param camera_matrix: [..., 3, 4]
    :param camera_intrinsics: [..., 3, 3]
    """
    rotation = torch.inverse(camera_matrix[..., :3])
    translation = -torch.matmul(rotation, camera_matrix[..., 3:])

    points3d = torch.matmul(rotation.unsqueeze(-3), points3d.unsqueeze(-1))
    points3d = points3d + translation.unsqueeze(-3)
    points3d = torch.matmul(camera_intrinsics.unsqueeze(-3).unsqueeze(-3), points3d)
    points3d = points3d.squeeze(-1)
    points3d = from_homogeneous(points3d)

    return points3d
