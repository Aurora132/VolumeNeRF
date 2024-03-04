import numpy as np
import torch
import torch.nn as nn
from .N_generator_element import *


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.step = 0


class UNet_generator(Generator):
    def __init__(self, input_nc, output_nc, ngf=64, bilinear=True, norm_layer=nn.BatchNorm2d):
        super(UNet_generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.bilinear = bilinear

        self.inc = DoubleConv(input_nc, ngf, norm_layer=norm_layer)
        self.down1 = Down(ngf, 2 * ngf, norm_layer=norm_layer)
        self.down2 = Down(2 * ngf, 4 * ngf, norm_layer=norm_layer)
        self.down3 = Down(4 * ngf, 8 * ngf, norm_layer=norm_layer)

        factor = 2 if bilinear else 1
        self.down4 = Down(8 * ngf, 16 * ngf // factor)
        self.up1 = Up(16 * ngf, 8 * ngf // factor, bilinear, norm_layer=norm_layer)
        self.up2 = Up(8 * ngf, 4 * ngf // factor, bilinear, norm_layer=norm_layer)
        self.up3 = Up(4 * ngf, 2 * ngf // factor, bilinear, norm_layer=norm_layer)
        self.up4 = Up(2 * ngf, ngf // factor, bilinear, norm_layer=norm_layer)

    def forward(self, x):
        res = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        res.append(x5)

        x = self.up1(x5, x4)
        res.append(x)
        x = self.up2(x, x3)
        res.append(x)
        x = self.up3(x, x2)
        res.append(x)
        x = self.up4(x, x1)
        res.append(x)
        return res


class GenerateRF(Generator):
    def __init__(self, spacing, sdr, del_size, input_dim=1, output_dim=1, hidden_dim=64, input_coor_dim=1,
                 atte_clamp_mode='sigmoid', norm_layer=nn.BatchNorm3d, norm_layer2d=nn.BatchNorm2d, depths=(3, 3, 9, 3), offset_points=3):
        super().__init__()
        dims = [hidden_dim // 2, hidden_dim, hidden_dim * 2, hidden_dim * 4, hidden_dim * 8]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.atte_clamp_mode = atte_clamp_mode

        self.spacing = spacing
        self.sdr = sdr
        self.del_size = del_size
        self.offset_points = offset_points

        self.in_tr = InputTransition(input_coor_dim, hidden_dim, hidden_dim, 1, norm_layer, norm_layer2d)
        self.down_1 = DownTransition(hidden_dim, 2 * hidden_dim, hidden_dim, 1, norm_layer, norm_layer2d)
        self.down_2 = DownTransition(2 * hidden_dim, 4 * hidden_dim, hidden_dim, 2, norm_layer, norm_layer2d)
        self.down_3 = DownTransition(4 * hidden_dim, 8 * hidden_dim, hidden_dim, 3, norm_layer, norm_layer2d)
        self.down_4 = DownTransition(8 * hidden_dim, 8 * hidden_dim, hidden_dim, 2, norm_layer, norm_layer2d)

        self.up_1 = UpTransition(16 * hidden_dim, 8 * hidden_dim, 2, spacing, sdr, del_size, offset_points,
                                 norm_layer)
        self.up_2 = UpTransition(8 * hidden_dim, 4 * hidden_dim, 2, spacing, sdr, del_size, offset_points,
                                 norm_layer)
        self.up_3 = UpTransition(4 * hidden_dim, 2 * hidden_dim, 1, spacing, sdr, del_size, offset_points,
                                 norm_layer)
        self.up_4 = UpTransition(2 * hidden_dim, 1 * hidden_dim, 1, spacing, sdr, del_size, offset_points,
                                 norm_layer)

        self.out = OutConv(hidden_dim, output_dim)

        self.mapping_network = ConvNeXt(in_chans=input_dim, depths=depths, dims=dims)

    def forward(self, input_coor, input_Xray):
        Xray_style = self.mapping_network(input_Xray)
        return self.forward_with_Xray(input_coor, Xray_style)

    def forward_with_Xray(self, input_coor, Xray_style, eps=1e-3):
        out32 = self.in_tr(input_coor, Xray_style[-5])
        out64 = self.down_1(out32, Xray_style[-4])
        out128 = self.down_2(out64, Xray_style[-3])
        out256 = self.down_3(out128, Xray_style[-2])
        out512 = self.down_4(out256, Xray_style[-1])
        out, _ = self.up_1(out512, out256, Xray_style[-1])
        out, _ = self.up_2(out, out128, Xray_style[-2])
        out, _ = self.up_3(out, out64, Xray_style[-3])
        out, final = self.up_4(out, out32, Xray_style[-4])
        out = self.out(out, Xray_style[-5])
        out = out.squeeze(dim=1)

        kernel_sharp = [[0., -1., 0.], [-1., 5., -1.], [0., -1., 0.]]
        kernel_sharp = torch.FloatTensor(kernel_sharp).unsqueeze(0).unsqueeze(0).repeat_interleave(out.shape[1], dim=0).cuda()
        weight_sharp = nn.Parameter(data=kernel_sharp, requires_grad=False)

        out = F.conv2d(out, weight_sharp, groups=out.shape[1], padding=1)
        out = torch.unsqueeze(out, dim=1)

        if self.atte_clamp_mode == 'sigmoid':
            out = torch.sigmoid(out)
        elif self.atte_clamp_mode == 'widen_sigmoid':
            out = torch.sigmoid(out) * (1 + 2 * eps) - eps

        return out, final


if __name__ == '__main__':

    input_Xray = torch.zeros((1, 2, 128, 128)).cuda()
    input_CT = torch.zeros((1, 1, 128, 128, 128)).cuda()

    net = GenerateRF(np.array([2.5, 2.5, 2.5]), 949 // 2, 5, hidden_dim=34, input_dim=2, offset_points=3).cuda()
    res, offset_coor = net(input_CT, input_Xray)
    print(res.shape, offset_coor.shape)
