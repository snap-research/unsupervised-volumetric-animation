"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import torch
from torch import nn
from torch.nn import BatchNorm2d
from torch.nn import functional as F


class ResBlock2d(nn.Module):
    """
    Residual block that preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size=3, padding=1):
        """
        :param in_features: number of features in the input
        :param kernel_size: size of the convolutional kernel
        :param padding: size of the padding
        """
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features,
                               out_channels=in_features,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features,
                               out_channels=in_features,
                               kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    2x up-sampling block (nearest upscale-conv-bn-relu) for the decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        """
        :param in_features: number of features in the input
        :param out_features: number of features in the output
        :param kernel_size: size of the convolutional kernel
        :param padding: size of the padding
        """
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    2x down-sampling block (conv-bn-relu-avg pool) for the encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        """
        :param in_features: number of features in the input
        :param out_features: number of features in the output
        :param kernel_size: size of the convolutional kernel
        :param padding: size of the padding
        """

        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple conv block (conv-bn-relu), that preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        """
        :param in_features: number of features in the input
        :param out_features: number of features in the output
        :param kernel_size: size of the convolutional kernel
        :param padding: size of the padding
        """

        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              padding=padding)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass (U-Net) encoder-decoder architecture consisting of UpBlock2d and DownBlock2d.
    Each block increase/decrease number of features 2 times up to max_features.
    """

    def __init__(self, in_features, block_expansion, num_blocks=3, max_features=256):
        """
        :param in_features: number of input features.
        :param block_expansion: the multiplier for the block size.
        :param num_blocks: number of blocks in the
        :param max_features: maximum number of features
        """

        super(Hourglass, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2**i)),
                            min(max_features, block_expansion * (2**(i + 1))),
                            kernel_size=3,
                            padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2**(i + 1)))
            out_filters = min(max_features, block_expansion * (2**i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        skips = [x]
        for down_block in self.down_blocks:
            skips.append(down_block(skips[-1]))

        out = skips.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = skips.pop()
            out = torch.cat([out, skip], dim=1)

        return out


class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited down-sampling, that down-sample images better.
    The class create a kernel for anti-aliasing in advance and then use it for all forwards.
    """

    def __init__(self, channels, scale):
        """
        :param channels: number of channels in the input
        :param scale: how much the image should be downscaled, e.g. 0.5 for 2 times downscale.
        """
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean)**2 / (2 * std**2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depth-wise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, inp, subsample=True):
        if self.scale == 1.0:
            return inp

        out = F.pad(inp, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        if subsample:
            out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class UpResBlock3d(nn.Module):
    """
    Residual 3d block that doubles spatial resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        """
        :param in_features: number of features in the input
        :param out_features: number of features in the output
        :param kernel_size: size of the convolutional kernel
        :param padding: size of the padding
        """
        super(UpResBlock3d, self).__init__()
        self.skip_conv = nn.Conv3d(in_features, out_features, 1)
        self.norm1 = nn.BatchNorm3d(in_features)
        self.norm2 = nn.BatchNorm3d(out_features)

        self.conv1 = nn.Conv3d(in_features, out_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_features, out_features, kernel_size=kernel_size, padding=padding)

    def forward(self, inp):
        inp = F.interpolate(inp, scale_factor=2, mode='nearest')
        res = self.skip_conv(inp)
        main = F.relu(inp)
        main = self.norm1(main)
        main = self.conv1(main)
        main = F.relu(main)
        main = self.norm2(main)
        main = self.conv2(main)
        return res + main


class SameResBlock3d(nn.Module):
    """
    Residual 3d block that doubles spatial resolution.
    """

    def __init__(self, in_features, kernel_size=3, padding=1):
        """
        :param in_features: number of features in the input
        :param kernel_size: size of the convolutional kernel
        :param padding: size of the padding
        """
        super(SameResBlock3d, self).__init__()
        self.norm1 = nn.BatchNorm3d(in_features)
        self.norm2 = nn.BatchNorm3d(in_features)

        self.conv1 = nn.Conv3d(in_features, in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_features, in_features, kernel_size=kernel_size, padding=padding)

    def forward(self, inp):
        main = F.relu(inp)
        main = self.norm1(main)
        main = self.conv1(main)
        main = F.relu(main)
        main = self.norm2(main)
        main = self.conv2(main)
        return inp + main


class AttnBlock3d(nn.Module):

    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.norm = torch.nn.BatchNorm3d(in_features)
        self.k = torch.nn.Conv3d(in_features, in_features, kernel_size=1)
        self.q = torch.nn.Conv3d(in_features, in_features, kernel_size=1)
        self.v = torch.nn.Conv3d(in_features, in_features, kernel_size=1)
        self.proj_out = torch.nn.Conv3d(in_features, in_features, kernel_size=1)

    def forward(self, inp):
        hiden = inp
        hiden = self.norm(hiden)
        q = self.q(hiden)
        k = self.k(hiden)
        v = self.v(hiden)

        # compute attention
        bs, ch, d, h, w = q.shape
        q = q.reshape(bs, ch, d * h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(bs, ch, d * h * w)
        attn = torch.bmm(q, k)
        attn = attn * (int(ch)**(-0.5))
        attn = torch.nn.functional.softmax(attn, dim=2)

        # attend to values
        v = v.reshape(bs, ch, d * h * w)
        attn = attn.permute(0, 2, 1)
        hiden = torch.bmm(v, attn)
        hiden = hiden.reshape(bs, ch, d, h, w)

        hiden = self.proj_out(hiden)

        return inp + hiden


class Decoder3d(nn.Module):
    """
    3d Decoder that takes an embedding and output voxel cube using series of UpResBlock3d.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 block_expansion=512,
                 num_blocks=4,
                 same_blocks=1,
                 attn_positions=None,
                 min_features=-1):
        """
        :param in_features: size of the embedding
        :param out_features: features in the final cube
        :param block_expansion: num features in the first block
        :param num_blocks: number of upscale blocks
        :param min_features: in each block number of features decrease twice, this is the minimal number of features
        """
        super(Decoder3d, self).__init__()

        self.fc = nn.Linear(in_features, (4**3) * block_expansion)
        num_features = block_expansion
        blocks = []

        for macro_block_idx in range(num_blocks):
            next_features = max(min_features, num_features // 2)
            blocks.append(UpResBlock3d(num_features, next_features))
            for i in range(same_blocks):
                blocks.append(SameResBlock3d(next_features))
            if (attn_positions is not None) and (macro_block_idx in attn_positions):
                blocks.append(AttnBlock3d(next_features))
            num_features = next_features

        self.main_body = nn.Sequential(*blocks)
        self.final_norm = nn.BatchNorm3d(num_features)
        self.final_conv = nn.Conv3d(num_features, out_features, 3, padding=1)

    def forward(self, inp):
        shape = [inp.shape[0], -1, 4, 4, 4]
        inp = self.fc(inp).view(*shape)
        inp = self.main_body(inp)
        inp = self.final_norm(inp)
        inp = self.final_conv(inp)

        return inp
