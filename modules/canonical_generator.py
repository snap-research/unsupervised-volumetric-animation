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
from torch import nn

from modules.blocks import Decoder3d
from modules.whitened_embedding import WhitenedEmbedding


class CanonicalGenerator(nn.Module):
    """
    Generate canonical voxel cubes and render them into the image.
    """

    def __init__(self,
                 n_parts,
                 max_videos,
                 rendering_features=3,
                 embedding_size=128,
                 block_expansion=512,
                 voxel_resolution=64,
                 same_blocks=0,
                 attn_positions=None,
                 perturb_z=True,
                 n_samples=64,
                 bg_plate_size=0.1,
                 bound_inc=1.075,
                 absorb_all_density=False,
                 noise_std=0.5,
                 random_density_flip=False,
                 noise_std_max_steps=100000):
        """
        :param n_parts: number of object parts

        Network parameters:
        :param max_videos: maximum number of videos (identities) in the dataset, (used for creating embedding)
        :param rendering_features: number of features to render for RGB it is 3
        :param embedding_size: size of the embedding for each identity
        :param block_expansion: size of the network
        :param voxel_resolution: size of the voxel grid

        Rendering parameters:
        :param perturb_z: sample point in the random position on the ray
        :param n_samples: number points per ray
        :param bg_plate_size: size of the background plate (0 for no plate).
        :param bound_inc: pretend that voxel cube is a bit larger than a scene (this is how much larger)
        :param absorb_all_density: last point in a ray absorb all density, so that there is no transparent regions
        :param noise_std: maximal amount of noise added to the density
        :param noise_std_max_steps: noise added to density is decreasing linearly, after this number of steps it is 0
        """
        super(CanonicalGenerator, self).__init__()
        self.embedding = WhitenedEmbedding(max_videos, embedding_size)
        self.decoder = Decoder3d(embedding_size,
                                 rendering_features + n_parts + 1,
                                 block_expansion=block_expansion,
                                 same_blocks=same_blocks,
                                 attn_positions=attn_positions,
                                 num_blocks=np.round(np.log2(voxel_resolution / 4)).astype(int))
        self.n_parts = n_parts
        self.voxel_resolution = voxel_resolution

        # Rendering parameters:
        self.perturb_z = perturb_z
        self.n_samples = n_samples
        self.bg_plate_size = bg_plate_size
        self.bound_inc = bound_inc
        self.absorb_all_density = absorb_all_density
        self.noise_std = noise_std
        self.noise_std_max_steps = noise_std_max_steps
        self.random_density_flip = random_density_flip

    def get_canonical_by_id(self, video_id):
        """
        :param video_id: [bs,] id numbers of the identities for which we extract canonical
        :return: [bs, n_parts + rendering_features + 1, voxel_h, voxel_w, voxel_d]
        """
        return self.decoder(self.embedding(video_id))

    def get_random_canonical(self, num_random=1):
        """
        :param num_random: number of random samples
        :param device: where to initialize embeddings, None for the same place where embeddings
        :return: canonical[num_random, n_parts + rendering_features + 1, voxel_h, voxel_w, voxel_d]
        """
        return self.decoder(self.embedding.get_random(num_random))

    def get_canonical_with_emb(self, emb):
        """
        :param emb: emb [bs, embedding_size]
        :return: canonical[bs, n_parts + rendering_features + 1, voxel_h, voxel_w, voxel_d]
        """
        return self.decoder(emb)

    def sample_canonical(self, fg_points, bg_points, canonical, current_training_step=-1):
        """
        Sample point from canonical cube.
        First sample lbs weights, then bend rays and sample actual radiance and density.

        :param fg_points: points from foreground rays [bs, n_parts, h, w, points_per_ray, 3]
        :param bg_points: points from background rays [bs, 1, h, w, points_per_ray, 3]
        :param canonical: canonical cube [bs, n_parts + rendering_features + 1, voxel_h, voxel_w, voxel_d]
        :param current_training_step: this is used to infer the amount of noise to add to the density

        :return: output dict with:
                 - fg_segmentations [bs, n_parts, h, w, points_per_ray]
                 - bg_segmentations [bs, 1, h, w, points_per_ray]
                 - radiance [bs, rendering_features, h, w, point_per_ray]
                 - density [bs, 1, h, w, point_per_ray]
                 - fg_density [bs, n_parts, h, w, point_per_ray]
                 - bg_density [bs, 1, h, w, points_per_ray]
                 - point_lbs [bs, n_parts, h, w, points_per_ray]
        """
        bs, n_parts, h, w, points_per_ray, _ = fg_points.shape
        voxel_h, voxel_w, voxel_d = self.voxel_resolution, self.voxel_resolution, self.voxel_resolution

        fg_points = fg_points.reshape(bs * n_parts, h, w, points_per_ray, 3)

        canonical_lbs = canonical[:, :self.n_parts]
        canonical_lbs = canonical_lbs.reshape(bs * n_parts, 1, voxel_h, voxel_w, voxel_d)

        canonical_fg = canonical[:, self.n_parts:]
        bg_density = torch.zeros_like(canonical[:, -1:])
        if self.bg_plate_size != 0:
            border = int(bg_density.shape[2] * self.bg_plate_size)
            bg_density[:, :, -border:] = 100
        canonical_bg = torch.cat([canonical[:, self.n_parts:-1], bg_density], axis=1)

        # bs * n_parts x 1 x h x w x points_per_ray
        point_lbs = F.grid_sample(canonical_lbs, fg_points, padding_mode='zeros', align_corners=True)
        point_lbs = point_lbs.view(bs, n_parts, h, w, points_per_ray)
        point_lbs = F.softmax(point_lbs, dim=1)

        fg_points = fg_points.view(bs, n_parts, h, w, points_per_ray, 3) * point_lbs.unsqueeze(-1)
        fg_points = fg_points.sum(axis=1)

        # bs x 4 x h x w x point_per_ray
        if self.random_density_flip:
            # generate random flip vector
            flip = torch.randint(0, 2, (bs, 1, 1, 1, 1), device=canonical_fg.device).float()
            # randomnly flip density vertically
            canonical_density = canonical_fg[:, -1:]  * (1 - flip) + canonical_fg[:, -1:].flip(-1) * flip
            canonical_fg = torch.cat([canonical_fg[:, :-1], canonical_density], axis=1)
                                            
        fg_features = F.grid_sample(canonical_fg, fg_points, padding_mode='zeros', align_corners=True)
        bg_features = F.grid_sample(canonical_bg, bg_points[:, 0], padding_mode='zeros', align_corners=True)

        fg_density = fg_features[:, -1:]
        fg_radiance = fg_features[:, :-1]
        bg_density = bg_features[:, -1:]
        bg_radiance = bg_features[:, :-1]

        if self.noise_std != 0 and self.training:
            current_noise_std = self.noise_std * max(0.0, 1 - current_training_step / self.noise_std_max_steps)
            fg_noise = torch.randn(fg_density.shape, device=fg_density.device) * current_noise_std
            bg_noise = torch.randn(bg_density.shape, device=fg_density.device) * current_noise_std
            part_density = point_lbs * F.relu(fg_density)
            fg_density = F.relu(fg_density + fg_noise)
            bg_density = F.relu(bg_density + bg_noise)
        else:
            fg_density = F.relu(fg_density)
            part_density = point_lbs * fg_density
            bg_density = F.relu(bg_density)

        join_density = fg_density + bg_density
        join_density_bounded = torch.clamp(join_density, min=1e-5)
        radiance = (fg_density * fg_radiance + bg_density * bg_radiance) / join_density_bounded

        out = {
            'fg_segmentations': point_lbs * fg_density / join_density_bounded,
            'bg_segmentations': bg_density / join_density_bounded,
            'radiance': radiance,
            'density': join_density.squeeze(1),
            'fg_density': fg_density,
            'bg_density': bg_density,
            'part_density': part_density,
            'fg_points': fg_points
        }

        return out

    def render(self,
               camera_matrix,
               bg_camera_matrix,
               canonical,
               camera_intrinsics,
               ray_grid,
               scene_bounds,
               compute_normals=False,
               current_training_step=-1):
        """
        Render the canonical into the image (or image part)

        :param camera_matrix: [bs, n_parts, 3, 4] - poses of all parts
        :param bg_camera_matrix: [bs, 1, 3, 4] - camera pose for background
        :param canonical: [bs, n_parts + rendering_features + 1, voxel_h, voxel_w, voxel_d] - canonical cube
        :param camera_intrinsics: [bs, 3, 3] - camera intrinsics matrix
        :param ray_grid: [bs, h, w, 2] - grid of pixels at which to sample rays (usually just a meshgrid h x w)
        :param scene_bounds: [2, 3] bound of the scene that is being rendered, i.e. min/max x, y and z.
        :param compute_normals: do we need to compute normals during rendering (this requires additional pass)
        :param current_training_step: this is used to infer the amount of noise to add to the density
        """
        fg_rays_origin, fg_rays_direction = self.get_rays(camera_matrix, camera_intrinsics, ray_grid)
        bg_rays_origin, bg_rays_direction = self.get_rays(bg_camera_matrix, camera_intrinsics, ray_grid)

        # Compute t_val: t_vals is limited by scene_bounds.
        # Each ray is intersected with bound and everything in between is equally spaced.
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=bg_rays_origin.device)
        shape = list(bg_rays_origin.shape[:-1]) + list(t_vals.shape)
        t_vals = t_vals.expand(shape)
        near, far = self.compute_t(bg_rays_origin, bg_rays_direction, scene_bounds)
        z_vals = near * (1. - t_vals) + far * t_vals

        if self.perturb_z and self.training:
            # get intervals between samples
            mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mid, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mid], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        # bs x n_parts x h x w points_per_ray x 3
        fg_points = fg_rays_origin[..., None, :] + fg_rays_direction[..., None, :] * z_vals[..., None]
        # bs x 1 x h x w points_per_ray x 3
        bg_points = bg_rays_origin[..., None, :] + bg_rays_direction[..., None, :] * z_vals[..., None]

        # Compute the size of the canonical cube, which is a bit larger (by self.bound_inc) than scene_bounds.
        scene_center = scene_bounds.mean(dim=0, keepdim=True)
        increased_bounds = self.bound_inc * (scene_bounds - scene_center) + scene_center
        fg_points = self.normalize_points(fg_points, increased_bounds)
        bg_points = self.normalize_points(bg_points, increased_bounds)

        out = self.sample_canonical(fg_points, bg_points, canonical, current_training_step=current_training_step)

        if compute_normals:
            normals = -torch.autograd.grad(out['fg_density'].sum(), out['fg_points'], create_graph=True)[0]
            normals = normals / (increased_bounds[1] - increased_bounds[0]).expand_as(normals)
            rotation = camera_matrix[:, :, None, None, None, :, :3]
            rotation = rotation * out['fg_segmentations'].unsqueeze(-1).unsqueeze(-1)
            rotation = rotation.sum(axis=1)
            rotation = rotation.permute(0, 1, 2, 3, 5, 4)
            normals = (rotation @ normals.unsqueeze(-1)).squeeze(-1)
            normals = F.normalize(normals, dim=-1)
            out['normals'] = normals.permute(0, 4, 1, 2, 3)

        out = self.raw2outputs(out, z_vals.squeeze(1))

        return out

    def raw2outputs(self, rendering_dict, z_vals):
        """
        Perform neural rendering integration

        :param rendering_dict: dict with:
                     - fg_segmentations [bs, n_parts, h, w, points_per_ray]
                     - bg_segmentations [bs, 1, h, w, points_per_ray]
                     - radiance [bs, rendering_features, h, w, point_per_ray]
                     - density [bs, 1, h, w, point_per_ray]
                     - fg_density [bs, n_parts, h, w, point_per_ray]
                     - bg_density [bs, 1, h, w, points_per_ray]
                     - point_lbs [bs, n_parts, h, w, points_per_ray]
                     - [optionally] normals [bs, 1, h, w, points_per_ray]
        :param z_vals: z_values of the points on the rays

        :return out: dict with:
                     - fg_segmentations [bs, n_parts, h, w]
                     - bg_segmentations [bs, 1, h, w]
                     - radiance [bs, rendering_features, h, w]
                     - depth [bs, 1, h, w]
                     - occupancy [bs, 1, h, w]
                     - fg_density [bs, n_parts, h, w, point_per_ray]
                     - bg_density [bs, 1, h, w, points_per_ray]
                     - point_lbs [bs, n_parts, h, w, points_per_ray]
                     - [optionally] normals [bs, 1, h, w]

        """
        radiance = rendering_dict['radiance']
        density = rendering_dict['density']

        dists = z_vals[..., 1:] - z_vals[..., :-1]

        if self.absorb_all_density:
            alpha = 1. - torch.exp(-density[..., :-1] * dists)
            alpha = torch.cat([alpha, torch.ones_like(alpha[..., :1])], axis=-1)
        else:
            dists = torch.cat([dists, dists[..., -1:].detach()], axis=-1)
            alpha = 1. - torch.exp(-density * dists)

        border = torch.ones_like(alpha[..., :1])
        weights = alpha * torch.cumprod(torch.cat([border, 1. - alpha + 1e-10], -1), -1)[..., :-1]

        radiance = torch.sum(weights[:, None] * radiance, -1)
        depth = torch.sum(weights * z_vals, -1)
        occupancy = torch.sum(weights, -1)

        fg_segmentation = torch.sum(weights[:, None] * rendering_dict['fg_segmentations'], -1)
        bg_segmentation = torch.sum(weights[:, None] * rendering_dict['bg_segmentations'], -1)

        result = {
            'fg_segmentation': fg_segmentation,
            'bg_segmentation': bg_segmentation,
            'radiance': radiance,
            'depth': depth.unsqueeze(1),
            'occupancy': occupancy.unsqueeze(1),
            'fg_density': rendering_dict['fg_density'],
            'bg_density': rendering_dict['bg_density'],
            'part_density': rendering_dict['part_density']
        }

        if 'normals' in rendering_dict:
            result['normals'] = torch.sum(weights[:, None] * rendering_dict['normals'], -1)

        return result

    @staticmethod
    def compute_t(rays_origin, rays_direction, bounds):
        near = bounds[0].expand(rays_origin.shape)
        far = bounds[1].expand(rays_origin.shape)

        t_near = (near - rays_origin) / rays_direction
        t_far = (far - rays_origin) / rays_direction

        tmin = torch.minimum(t_near, t_far)
        tmax = torch.maximum(t_near, t_far)

        tmin = torch.max(tmin, axis=-1, keepdim=True)[0]
        tmax = torch.min(tmax, axis=-1, keepdim=True)[0]

        return tmin, tmax

    @staticmethod
    def get_rays(camera_matrix, camera_intrinsics, ray_grid):
        i = ray_grid[..., 0]
        j = ray_grid[..., 1]

        i = i.unsqueeze(1).repeat(1, camera_matrix.shape[1], 1, 1)
        j = j.unsqueeze(1).repeat(1, camera_matrix.shape[1], 1, 1)

        # bs x n_parts x h x w x 3
        directions = torch.stack([(i - camera_intrinsics[:, None, 0:1, 2:]) / camera_intrinsics[:, None, 0:1, 0:1],
                                  (j - camera_intrinsics[:, None, 1:2, 2:]) / camera_intrinsics[:, None, 1:2, 1:2],
                                  torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_directions = torch.matmul(camera_matrix[:, :, None, None, :3, :3], directions.unsqueeze(-1))
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_directions = rays_directions.squeeze(-1)

        # bs x n_parts x h x w x 3
        rays_origin = camera_matrix[:, :, None, None, :3, -1].expand(rays_directions.shape)
        rays_directions = rays_directions / torch.norm(rays_directions, dim=-1, keepdim=True)

        return rays_origin, rays_directions

    @staticmethod
    def normalize_points(points, bounds):
        """
        Normalize from point from bounds range to [-1, 1]

        :param points: [bs, n_parts, h, w, point_per_ray, 3]
        :param bounds: [2, 3]
        :return: normalized points
        """
        lower = bounds[0].expand_as(points)
        upper = bounds[1].expand_as(points)
        return 2 * (points - lower) / (upper - lower) - 1
