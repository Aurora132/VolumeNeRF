import numpy as np
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from .Deform_Attention import *


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.BatchNorm3d):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            DoubleConv(out_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm_layer=nn.BatchNorm2d):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_layer=norm_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_layer=norm_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


def passthrough(x, **kwargs):
    return x


class LUConv(nn.Module):
    def __init__(self, channels, in_channels=None, norm_layer=nn.BatchNorm3d):
        super(LUConv, self).__init__()
        if not in_channels:
            self.conv1 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = norm_layer(channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(channels, depth, norm_layer, in_channels=None):
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(LUConv(channels, in_channels=in_channels, norm_layer=norm_layer))
        else:
            layers.append(LUConv(channels, norm_layer=norm_layer))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, nConvs, norm_layer=nn.BatchNorm3d, norm_layer2d=nn.BatchNorm2d):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn1 = norm_layer(out_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.ops = _make_nConv(out_channels, nConvs, norm_layer)
        D_size = 128

        compress_block_1D = [
            nn.Conv2d(hidden_dim // 2, D_size, kernel_size=3, padding=1, bias=False),
            norm_layer2d(D_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ]
        self.compress_block_1D = nn.Sequential(*compress_block_1D)

        compress_block_2D = [
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.compress_block_2D = nn.Sequential(*compress_block_2D)

        conv_block1 = [
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.conv_block1 = nn.Sequential(*conv_block1)

        conv_block2 = [
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.conv_block2 = nn.Sequential(*conv_block2)

    def forward(self, x, xray):
        B, C, D, H, W = x.shape
        down = self.relu1(self.bn1(self.conv1(x)))
        xray_1D = nn.Softmax(dim=1)(self.compress_block_1D(xray)).unsqueeze(1)
        xray_2D = self.compress_block_2D(xray).unsqueeze(2).expand(B, -1, D, H, W)
        xray = self.conv_block1(xray_1D * xray_2D)
        out = torch.cat((down, xray), 1)
        out = self.conv_block2(out)
        out = self.ops(out)
        return out


class DownTransition(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, nConvs, norm_layer=nn.BatchNorm3d, norm_layer2d=nn.BatchNorm2d, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv = DoubleConv3D(out_channels, out_channels, norm_layer=norm_layer)
        self.bn1 = norm_layer(out_channels)
        self.do1 = passthrough
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(out_channels, nConvs, norm_layer)

        D_size = hidden_dim * 64 // in_channels
        compress_block_1D = [
            nn.Conv2d(in_channels, D_size, kernel_size=3, padding=1, bias=False),
            norm_layer2d(D_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        ]
        self.compress_block_1D = nn.Sequential(*compress_block_1D)

        compress_block_2D = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.compress_block_2D = nn.Sequential(*compress_block_2D)

        conv_block1 = [
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.conv_block1 = nn.Sequential(*conv_block1)

        conv_block2 = [
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.conv_block2 = nn.Sequential(*conv_block2)

    def forward(self, x, xray):
        down = self.relu(self.bn1(self.down_conv(x)))
        B, C, D, H, W = down.shape
        out = self.conv(down)
        xray_1D = nn.Softmax(dim=1)(self.compress_block_1D(xray)).unsqueeze(1)
        xray_2D = self.compress_block_2D(xray).unsqueeze(2).expand(B, C, D, H, W)
        xray = self.conv_block1(xray_1D * xray_2D)
        out = torch.cat((out, xray), 1)
        out = self.conv_block2(out)
        out = self.ops(out)
        return out


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, nConvs, spacing, sdr, del_size, offset_points,
                 dropout=False, norm_layer=nn.BatchNorm3d):
        super(UpTransition, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3D(in_channels // 2, out_channels // 2, norm_layer=norm_layer)

        self.up_conv = nn.Conv3d(in_channels * 3 // 4, out_channels // 2, kernel_size=3, padding=1)
        self.bn1 = norm_layer(out_channels // 2)
        self.do = passthrough
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if dropout:
            self.do = nn.Dropout3d()
        self.ops = _make_nConv(out_channels // 2, nConvs, norm_layer)
        self.DeformAttention = DeformAttention(in_channels // 2, spacing, sdr, del_size, offset_points, norm_layer=norm_layer)

    def forward(self, x, skipx, xray):
        B, C, D, H, W = x.shape
        x, _ = self.DeformAttention(x, xray)
        out = self.do(x)
        out = self.up(out)
        out = self.conv(out)
        xcat = torch.cat((out, skipx), 1)
        xcat = self.relu(self.bn1(self.up_conv(xcat)))
        out = self.ops(xcat)
        return out, _


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, xray):
        xray = xray.unsqueeze(3).expand_as(x)
        return self.conv(torch.concat((x, xray), dim=1))


class Block(nn.Module):

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + x
        return x


class ConvNeXt(nn.Module):
    def __init__(self, in_chans: int = 1, out_chans: int = 1, depths: list = None, dims: list = None,
                 layer_scale_init_value: float = 1e-6, drop_rate: float = 0., norm_layer=nn.BatchNorm2d, bilinear=True):
        super().__init__()
        self.first_layer = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=3, padding=1),
                                         nn.BatchNorm2d(dims[0]))
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
                             nn.BatchNorm2d(dims[1]))
        self.downsample_layers.append(stem)

        for i in range(1, 4):
            downsample_layer = nn.Sequential(nn.BatchNorm2d(dims[i]),
                                             nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        for i in range(1, 5):
            stage = nn.Sequential(
                *[Block(dim=dims[i], layer_scale_init_value=layer_scale_init_value, drop_rate=drop_rate)
                  for _ in range(depths[i - 1])])
            self.stages.append(stage)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)

    def forward(self, x: torch.Tensor):
        xx_skip = []
        x = self.first_layer(x)
        xx_skip.append(x)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            xx_skip.append(x)
        return xx_skip


def convnext_tiny(in_chans=1, depths=(3, 3, 9, 3), dims=(32, 64, 128, 256, 512)):
    model = ConvNeXt(in_chans=in_chans, depths=depths, dims=dims)
    return model


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = ConvNeXt(in_chans=1, out_chans=1, depths=[3, 3, 9, 3], dims=[32, 64, 128, 256, 512]).cuda()
    x = torch.ones((1, 1, 128, 128)).cuda()
    res = net(x)
    res_shape = [_.shape for _ in res]
    print(res_shape)

    total_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('G: Total ', total_num)
