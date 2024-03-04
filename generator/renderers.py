import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .N_generators import GenerateRF


def render_Xray(volume, step_length, idxs, target, source, drr_height, drr_weight=None, eps=1e-8):
    if drr_weight is None:
        drr_weight = drr_height
    batch_size, H, W, D = volume.shape
    ele_all = H * W * D
    idxs_all = torch.concat([idxs + i * ele_all for i in range(batch_size)], dim=0)
    voxels = torch.take(volume.flip([1]), idxs_all)
    weighted_voxels = voxels * step_length
    drr = torch.sum(weighted_voxels, dim=-1)
    raylength = (target - source + eps).norm(dim=-1)
    drr *= raylength
    return drr.reshape(-1, 1, drr_height, drr_weight)


class Generator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.step = 0


class Renderer(Generator):
    def __init__(self, step_length_path, idxs_path, target_path, source_path, coor_path, mean_path, drr_height, device,
                 representation_kwargs, drr_weight=None):
        super().__init__()
        self.step_length = np.load(step_length_path).astype(np.float32)
        self.idxs = np.load(idxs_path)
        self.target = np.load(target_path).astype(np.float32)
        self.source = np.load(source_path).astype(np.float32)
        self.coor3d = np.load(coor_path).astype(np.float32)
        self.mean_CT = np.load(mean_path).astype(np.float32)

        self.step_length = nn.Parameter(data=torch.from_numpy(self.step_length).to(device), requires_grad=False)
        self.idxs = nn.Parameter(data=torch.from_numpy(self.idxs).to(device), requires_grad=False)
        self.target = nn.Parameter(data=torch.from_numpy(self.target).to(device), requires_grad=False)
        self.source = nn.Parameter(data=torch.from_numpy(self.source).to(device), requires_grad=False)
        self.coor3d = nn.Parameter(data=torch.from_numpy(self.coor3d).to(device), requires_grad=False)
        self.mean_CT = nn.Parameter(data=torch.unsqueeze(torch.from_numpy(self.mean_CT).to(device), dim=0
                                                         ), requires_grad=False)

        self.drr_height = drr_height
        self.drr_weight = drr_weight
        self.representation = GenerateRF(**representation_kwargs)

    def forward(self, input_xray, prob_xray, eps=1e-8):

        device_type = input_xray.device
        batch_size, _, _, _ = input_xray.size()
        input_mean_CT = self.mean_CT.repeat(batch_size, 1, 1, 1, 1)

        net = self.representation.to(device_type)

        volume, final_coor = net(input_mean_CT, torch.concat((input_xray, prob_xray), dim=1))
        render_volume = torch.squeeze(volume, dim=1).permute(0, 2, 3, 1)

        drr = render_Xray(render_volume, self.step_length, self.idxs, self.target, self.source, self.drr_height,
                          self.drr_weight, eps=eps)

        torch.cuda.empty_cache()

        return volume, drr, final_coor


if __name__ == '__main__':
    import os

    root_path = '../DRR_Parameters'
    step_length_path = os.path.join(root_path, 'step_length.npy')
    idxs_path = os.path.join(root_path, 'idxs.npy')
    target_path = os.path.join(root_path, 'target.npy')
    source_path = os.path.join(root_path, 'source.npy')
    coor_path = os.path.join(root_path, 'coords_3D.npy')
    mean_path = os.path.join(root_path, 'mean_CT.npy')
    drr_height = 128
    representation_kwargs = {'hidden_dim': 4, 'input_coor_dim': 1, 'input_dim': 2, 'spacing': np.array([2.5, 2.5, 2.5]),
                             'sdr': 949 // 2, 'del_size': 5, 'offset_points': 3}

    renderer = Renderer(step_length_path, idxs_path, target_path, source_path, coor_path, mean_path, drr_height, 'cuda',
                        representation_kwargs)
    input_image = torch.ones((2, 1, 128, 128), dtype=torch.float32, requires_grad=True)
    input_xray = input_image.cuda()
    prob_xray = torch.ones((2, 1, 128, 128), dtype=torch.float32, requires_grad=True)
    prob_xray = prob_xray.cuda()
    input_CT = torch.ones((2, 1, 128, 128, 128), dtype=torch.float32, requires_grad=True)
    input_CT = input_CT.cuda()

    volume, drr, final_coor = renderer(input_xray, prob_xray)
    print(volume.requires_grad, drr.requires_grad)
    print(volume.shape, drr.shape, final_coor.shape)