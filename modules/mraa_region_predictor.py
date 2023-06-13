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
from torch import nn

from modules.blocks import Hourglass, AntiAliasInterpolation2d


class MRAARegionPredictor(nn.Module):
    """
    Region predictor from MRAA, which is used for mask prediction.
    """

    def __init__(self,
                 num_regions=10,
                 block_expansion=32,
                 max_features=1024,
                 num_blocks=5,
                 temperature=0.1,
                 scale_factor=0.25,
                 num_channels=3,
                 multipart=False,
                 threshold=0.0001):
        """
        :param num_regions: number of regions (should correspond to number of parts)
        :param block_expansion: size of the first conv block
        :param max_features: maximum number of features
        :param num_blocks: number of blocks in the network
        :param temperature: softmax temperature for regions
        :param num_channels: number of channels in the frame
        :param multipart: return mask with multiple parts or single part
        :param threshold: threshold for quantize soft mask to hard mask
        :param scale_factor: resize the image by this factor, before input
        """
        super(MRAARegionPredictor, self).__init__()
        self.predictor = Hourglass(block_expansion=block_expansion,
                                   in_features=num_channels,
                                   max_features=max_features,
                                   num_blocks=num_blocks)

        self.regions = nn.Conv2d(in_channels=self.predictor.out_filters,
                                 out_channels=num_regions,
                                 kernel_size=(7, 7),
                                 padding=3)

        self.temperature = temperature
        self.scale_factor = scale_factor
        self.multipart = multipart
        self.threshold = threshold

        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

        for param in self.parameters():
            param.requires_grad = False

    def load_state_dict(self, state_dict, strict=False):
        """
        Load state dict, but rename properly and don't load AntiAliasInterpolation2d parameters.
        """
        with torch.no_grad():
            self_state = self.state_dict()
            for name, param in state_dict.items():
                if name.startswith('down'):
                    continue
                name = name.replace('encoder.', '').replace('decoder.', '')
                if isinstance(param, nn.Parameter):
                    param = param.data
                self_state[name].copy_(param)

    def convert_regions_to_segmentation_mask(self, regions):
        """
        Convert soft regions to hard segmentation mask with background.

        :param regions: soft segmentation mask[bs, n_parts, h, w]
        :return: hard segmentation mask[bs, n_parts, h, w]
        """
        hard_mask, max_index = regions.max(dim=1, keepdim=True)
        hard_mask = hard_mask > self.threshold
        hard_mask = hard_mask.float()

        if self.multipart:
            num_classes = regions.shape[1]
            regions = torch.argmax(regions, dim=1)
            regions = F.one_hot(regions, num_classes=num_classes)
            regions = torch.permute(regions, (0, 3, 1, 2))
            result_mask = torch.cat([hard_mask * regions, (1 - hard_mask)], dim=1)
        else:
            result_mask = torch.cat([hard_mask, (1 - hard_mask)], dim=1)

        class_weights = 1 / torch.clamp(result_mask.mean(axis=(0, 2, 3)), min=1e-3)
        return result_mask, class_weights

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.regions(feature_map)

        final_shape = prediction.shape
        region = prediction.view(final_shape[0], final_shape[1], -1)
        region = F.softmax(region / self.temperature, dim=2)
        region = region.view(*final_shape)

        return self.convert_regions_to_segmentation_mask(region)
