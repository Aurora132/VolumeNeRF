import torch
import torch.nn as nn
import torch.nn.functional as F


def Gradient_edge(CT):
    CT = CT.squeeze(dim=1)

    kernel_x = [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).repeat_interleave(CT.shape[1], dim=0).cuda()
    kernel_y = [[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).repeat_interleave(CT.shape[1], dim=0).cuda()
    kernel_45 = [[-10., -3., 0.], [-3., 0., 3.], [0., 3., 10.]]
    kernel_45 = torch.FloatTensor(kernel_45).unsqueeze(0).unsqueeze(0).repeat_interleave(CT.shape[1], dim=0).cuda()
    kernel_135 = [[0., 3., 10.], [-3., 0., 3.], [-10., -3., 0.]]
    kernel_135 = torch.FloatTensor(kernel_135).unsqueeze(0).unsqueeze(0).repeat_interleave(CT.shape[1], dim=0).cuda()

    weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
    weight_45 = nn.Parameter(data=kernel_45, requires_grad=False)
    weight_135 = nn.Parameter(data=kernel_135, requires_grad=False)

    grad_x = F.conv2d(CT, weight_x, groups=CT.shape[1])
    grad_x = (grad_x - torch.min(grad_x)) / (torch.max(grad_x) - torch.min(grad_x))
    grad_y = F.conv2d(CT, weight_y, groups=CT.shape[1])
    grad_y = (grad_y - torch.min(grad_y)) / (torch.max(grad_y) - torch.min(grad_y))
    grad_45 = F.conv2d(CT, weight_45, groups=CT.shape[1])
    grad_45 = (grad_45 - torch.min(grad_45)) / (torch.max(grad_45) - torch.min(grad_45))
    grad_135 = F.conv2d(CT, weight_135, groups=CT.shape[1])
    grad_135 = (grad_135 - torch.min(grad_135)) / (torch.max(grad_135) - torch.min(grad_135))

    return torch.abs(grad_x), torch.abs(grad_y), torch.abs(grad_45), torch.abs(grad_135)
