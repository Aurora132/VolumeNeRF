import torch
import torch.nn as nn
from torch.nn.functional import normalize
import numpy as np


class DeformAttention(nn.Module):
    def __init__(self, inc, spacing, sdr, del_size, offset_points=3, kernel_size=3, padding=1, stride=1, norm_layer=nn.BatchNorm3d):
        super(DeformAttention, self).__init__()
        self.spacing = spacing
        self.sdr = sdr
        self.del_size = del_size
        self.offset_points = offset_points

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # p_conv: offset  m_conv: attention score
        self.p_conv = nn.Conv3d(2 * inc, 2 * offset_points, kernel_size=kernel_size, padding=padding, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.m_conv = nn.Conv3d(2 * inc, offset_points, kernel_size=kernel_size, padding=padding, stride=stride)
        nn.init.constant_(self.m_conv.weight, 0)
        self.m_conv.register_full_backward_hook(self._set_lr)

        self.conv1 = nn.Conv3d(inc * 2, inc, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv3d(inc, inc, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = norm_layer(inc)
        self.relu = nn.LeakyReLU(0.2)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, CT, Xray):
        # (B, 2C, D, H, W)  (B, D, H, W, 2)
        B, C, D, H, W = CT.shape
        CT_mix, X_ray_coor = self._get_initial_xray(self.spacing, CT, Xray, self.sdr, self.del_size)

        Xray = self.zero_padding(Xray)

        # (B, 2*offset_points, D, H, W)
        offset = self.p_conv(CT_mix)
        # (B, offset_points, D, H, W)
        m = torch.softmax(self.m_conv(CT_mix), dim=1)

        # (B, D, H, W, 2*offset_points)
        offset_coor = offset.permute(0, 2, 3, 4, 1)
        p_coor = torch.concat([offset_coor[..., :self.offset_points] + X_ray_coor[..., :1] + 1,
                               offset_coor[..., self.offset_points:] + X_ray_coor[..., 1:] + 1], dim=-1)
        p_coor = torch.cat([torch.clamp(p_coor[..., :self.offset_points], 0, Xray.size(2) - 1),
                            torch.clamp(p_coor[..., self.offset_points:], 0, Xray.size(3) - 1)], dim=-1)

        # (B, C, D, H, W, offset_points)
        Xray_offset = self.bilinear_interpolation_multi(p_coor, Xray)

        # (B, D, H, W, offset_points)
        m = m.contiguous().permute(0, 2, 3, 4, 1)
        # (B, 1, D, H, W, offset_points)
        m = m.unsqueeze(dim=1)
        # (B, C, D, H, W, offset_points)
        m = torch.cat([m for _ in range(Xray.size(1))], dim=1)
        Xray_offset *= m
        # (B, C, D, H, W)
        Xray_offset = torch.sum(Xray_offset, dim=-1)

        # (B, 2C, D, H, W)
        final_CT_mix = torch.cat([CT, Xray_offset], dim=1)
        # (B, C, D, H, W)
        out = self.relu(self.bn(self.conv1(final_CT_mix)))

        return self.conv2(out), p_coor

    def Rx(self, gamma, batch_size, device):
        t0 = torch.zeros(batch_size, 1).to(device)
        t1 = torch.ones(batch_size, 1).to(device)
        return torch.stack(
            [
                t1,
                t0,
                t0,
                t0,
                torch.cos(gamma.unsqueeze(1)),
                -torch.sin(gamma.unsqueeze(1)),
                t0,
                torch.sin(gamma.unsqueeze(1)),
                torch.cos(gamma.unsqueeze(1)),
            ],
            dim=1,
        ).reshape(batch_size, 3, 3)

    def Ry(self, phi, batch_size, device):
        t0 = torch.zeros(batch_size, 1).to(device)
        t1 = torch.ones(batch_size, 1).to(device)
        return torch.stack(
            [
                torch.cos(phi.unsqueeze(1)),
                t0,
                torch.sin(phi.unsqueeze(1)),
                t0,
                t1,
                t0,
                -torch.sin(phi.unsqueeze(1)),
                t0,
                torch.cos(phi.unsqueeze(1)),
            ],
            dim=1,
        ).reshape(batch_size, 3, 3)

    def Rz(self, theta, batch_size, device):
        t0 = torch.zeros(batch_size, 1).to(device)
        t1 = torch.ones(batch_size, 1).to(device)
        return torch.stack(
            [
                torch.cos(theta.unsqueeze(1)),
                -torch.sin(theta.unsqueeze(1)),
                t0,
                torch.sin(theta.unsqueeze(1)),
                torch.cos(theta.unsqueeze(1)),
                t0,
                t0,
                t0,
                t1,
            ],
            dim=1,
        ).reshape(batch_size, 3, 3)

    def _get_x(self, x, q):
        B, D, H, W, _ = q.size()
        _, C, Height, Weight = x.size()
        # (B, C, Height * Weight)
        x = x.contiguous().view(B, C, -1)

        # (B, D, H, W)
        index = q[..., 0] * Weight + q[..., 1]  # line * Weight + column
        # (B, C, D * H * W)
        index = index.contiguous().unsqueeze(1).expand(-1, C, -1, -1, -1).contiguous().view(B, C, -1)

        x_q = x.gather(dim=-1, index=index).contiguous().view(B, C, D, H, W)
        return x_q

    def bilinear_interpolation(self, p, x):
        # p: interpolation coordinate  x:extracted feature
        # p: (B, D, H, W, 2)  x: (B, C, H, W)
        B, C, Height, Weight = x.shape
        # p.detach().floor()
        q_lt = p.floor()
        q_rb = q_lt + 1

        # q: (B, D, H, W, 2)
        q_lt = torch.stack([torch.clamp(q_lt[..., 0], 0, Height - 1), torch.clamp(q_lt[..., 1], 0, Weight - 1)],
                           dim=-1).long()
        q_rb = torch.stack([torch.clamp(q_rb[..., 0], 0, Height - 1), torch.clamp(q_rb[..., 1], 0, Weight - 1)],
                           dim=-1).long()
        q_rt = torch.stack([torch.clamp(q_lt[..., 0], 0, Height - 1), torch.clamp(q_rb[..., 1], 0, Weight - 1)],
                           dim=-1).long()
        q_lb = torch.stack([torch.clamp(q_rb[..., 0], 0, Height - 1), torch.clamp(q_lt[..., 1], 0, Weight - 1)],
                           dim=-1).long()

        # interpolation weight g: (B, D, H, W)
        g_lt = (1 + (q_lt[..., 0].type_as(p) - p[..., 0])) * (1 + (q_lt[..., 1].type_as(p) - p[..., 1]))
        g_rb = (1 - (q_rb[..., 0].type_as(p) - p[..., 0])) * (1 - (q_rb[..., 1].type_as(p) - p[..., 1]))
        g_rt = (1 + (q_rt[..., 0].type_as(p) - p[..., 0])) * (1 - (q_rt[..., 1].type_as(p) - p[..., 1]))
        g_lb = (1 - (q_lb[..., 0].type_as(p) - p[..., 0])) * (1 + (q_lb[..., 1].type_as(p) - p[..., 1]))

        # interpolation pixel x_q: (B, C, D, H, W)
        x_q_lt = self._get_x(x, q_lt)
        x_q_rb = self._get_x(x, q_rb)
        x_q_rt = self._get_x(x, q_rt)
        x_q_lb = self._get_x(x, q_lb)

        # bilinear interpolation
        x_interpolation = g_lt.unsqueeze(dim=1) * x_q_lt + \
                          g_rb.unsqueeze(dim=1) * x_q_rb + \
                          g_rt.unsqueeze(dim=1) * x_q_rt + \
                          g_lb.unsqueeze(dim=1) * x_q_lb
        # x_interpolation: (B, C, D, H, W)
        return x_interpolation

    def _get_initial_xray(self, spacing, CT, Xray, sdr, del_size):
        B, C, D, H, W = CT.shape
        _, _, Height, Weight = Xray.shape
        device = CT

        rotations = torch.tensor([[torch.pi, 0.0, torch.pi / 2]]).expand(B, -1).to(CT)
        translations = (torch.tensor([D, H, W]) * torch.tensor(spacing) / 2).to(CT)
        translations = translations.unsqueeze(0).expand(B, -1)

        theta, phi, gamma = rotations[:, 0], rotations[:, 1], rotations[:, 2]
        R_z = self.Rz(theta, B, device)
        R_y = self.Ry(phi, B, device)
        R_x = self.Rx(gamma, B, device)
        Rxyz = torch.einsum("bij,bjk,bkl->bil", R_z, R_y, R_x)
        R = sdr * Rxyz

        source = R[..., 0].unsqueeze(1)
        center = -source
        source += translations.unsqueeze(1)
        center += translations.unsqueeze(1)

        R_ = normalize(R.clone(), dim=-1)
        u, v = R_[..., 1], R_[..., 2]
        basis = torch.stack([u, v], dim=1)

        h_off = 1.0 if Height % 2 else 0.5
        w_off = 1.0 if Weight % 2 else 0.5

        t = (torch.arange(-Height // 2, Height // 2) + h_off) * del_size
        s = (torch.arange(-Weight // 2, Weight // 2) + w_off) * del_size

        # target:coordinate of Xray → (B, H*W, 3)
        coefs = torch.cartesian_prod(t, s).reshape(-1, 2).to(CT)
        target = torch.einsum("bcd,nc->bnd", basis, coefs)
        target += center

        target_x = torch.arange(0, H) * spacing[0]
        target_y = torch.arange(0, W) * spacing[1]
        target_z = torch.arange(0, D) * spacing[2]

        # target:coordinate of CT → (B, D*H*W, 3)
        p_z, p_x, p_y = torch.meshgrid(target_z, target_x, target_y, indexing='ij')
        p_n = torch.cat([p_z.reshape(-1, 1), p_x.reshape(-1, 1), p_y.reshape(-1, 1)], 1).to(CT)
        target_CT = p_n.expand(B, -1, -1)

        # calculate paired CT and Xray
        # X_ray_coor → (B, D*H*W, 3)
        weight = ((center[:, :, 0] - source[:, :, 0]) / (target_CT[:, :, 0] - source[:, :, 0])).unsqueeze(2)
        X_ray_coor = weight * (target_CT - source) + source
        target_min = torch.min(target, dim=1, keepdim=True)[0].expand(-1, D * H * W, -1)
        target_max = torch.max(target, dim=1, keepdim=True)[0].expand(-1, D * H * W, -1)
        X_ray_coor = torch.clamp(X_ray_coor, target_min, target_max)

        # convert coordinate to index
        xray_delta = torch.tensor([[del_size, del_size]]).to(CT)
        delta = torch.einsum("bcd,nc->bnd", basis, xray_delta)
        X_ray_coor = (X_ray_coor - target_min) / delta
        X_ray_coor = X_ray_coor[..., 1:].reshape(B, D, H, W, 2)
        x_interpolation = self.bilinear_interpolation(X_ray_coor, Xray)

        # concat CT and Xray feature
        x_mix = torch.concat([CT, x_interpolation], dim=1)
        return x_mix, X_ray_coor

    def _get_x_multi(self, x, q):
        B, D, H, W, offset_points_double = q.size()
        _, C, Height, Weight = x.size()
        offset_points = offset_points_double // 2
        # (B, C, Height * Weight)
        x = x.contiguous().view(B, C, -1)

        # (B, D, H, W, offset_points)
        index = q[..., :offset_points] * Weight + q[..., offset_points:]  # line * Weight + column
        # (B, C, D * H * W * offset_points)
        index = index.contiguous().unsqueeze(1).expand(-1, C, -1, -1, -1, -1).contiguous().view(B, C, -1)

        x_q = x.gather(dim=-1, index=index).contiguous().view(B, C, D, H, W, offset_points)
        return x_q

    def bilinear_interpolation_multi(self, p, x):
        # p: interpolation coordinate  x:extracted feature
        # p: (B, D, H, W, 2k)  x: (B, C, H, W)
        B, C, Height, Weight = x.shape
        offset_points = p.size(-1) // 2
        # p.detach().floor()
        q_lt = p.floor()
        q_rb = q_lt + 1

        # q: (B, D, H, W, 2*offset_points)
        q_lt = torch.cat([torch.clamp(q_lt[..., :offset_points], 0, Height - 1),
                          torch.clamp(q_lt[..., offset_points:], 0, Weight - 1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :offset_points], 0, Height - 1),
                          torch.clamp(q_rb[..., offset_points:], 0, Weight - 1)], dim=-1).long()
        q_rt = torch.cat([torch.clamp(q_lt[..., :offset_points], 0, Height - 1),
                          torch.clamp(q_rb[..., offset_points:], 0, Weight - 1)], dim=-1).long()
        q_lb = torch.cat([torch.clamp(q_rb[..., :offset_points], 0, Height - 1),
                          torch.clamp(q_lt[..., offset_points:], 0, Weight - 1)], dim=-1).long()

        # interpolation weight g: (B, D, H, W, offset_points)
        g_lt = (1 + (q_lt[..., :offset_points].type_as(p) - p[..., :offset_points])) * (
                1 + (q_lt[..., offset_points:].type_as(p) - p[..., offset_points:]))
        g_rb = (1 - (q_rb[..., :offset_points].type_as(p) - p[..., :offset_points])) * (
                1 - (q_rb[..., offset_points:].type_as(p) - p[..., offset_points:]))
        g_rt = (1 + (q_rt[..., :offset_points].type_as(p) - p[..., :offset_points])) * (
                1 - (q_rt[..., offset_points:].type_as(p) - p[..., offset_points:]))
        g_lb = (1 - (q_lb[..., :offset_points].type_as(p) - p[..., :offset_points])) * (
                1 + (q_lb[..., offset_points:].type_as(p) - p[..., offset_points:]))

        # interpolation pixel x_q: (B, C, D, H, W, offset_points)
        x_q_lt = self._get_x_multi(x, q_lt)
        x_q_rb = self._get_x_multi(x, q_rb)
        x_q_rt = self._get_x_multi(x, q_rt)
        x_q_lb = self._get_x_multi(x, q_lb)

        # bilinear interpolation
        x_interpolation = g_lt.unsqueeze(dim=1) * x_q_lt + \
                          g_rb.unsqueeze(dim=1) * x_q_rb + \
                          g_rt.unsqueeze(dim=1) * x_q_rt + \
                          g_lb.unsqueeze(dim=1) * x_q_lb
        # x_interpolation: (B, C, D, H, W, offset_points)
        return x_interpolation


if __name__ == '__main__':
    C = 256
    H = 8
    net = DeformAttention(C, np.array([2.5, 2.5, 2.5]), 949 // 2, 5).cuda()
    Xray = torch.zeros([2, C, H, H]).cuda()
    CT = torch.zeros([2, C, H, H, H]).cuda()
    out, _ = net(CT, Xray)
    print(out.shape, out.requires_grad)
    print(out)

    loss = nn.MSELoss()(out, torch.ones_like(out).cuda())
    loss.backward()

    total_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('G: Total ', total_num)
    grad_num = sum(torch.sum(p.grad != 0).item() for p in net.parameters() if p.requires_grad)
    print('G: Total ', total_num, 'Grad is not zero ', grad_num)
