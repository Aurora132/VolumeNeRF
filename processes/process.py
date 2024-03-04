import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from loss.edge_loss import Gradient_edge


def negative_likelihood(image, mean, std):
    return torch.log(std) + (image - mean) ** 2 / (2 * (std ** 2))


class Pre3DProcess:
    def __init__(self, CT_edge_lambda, nerf_lambda):
        self.CT_edge_lambda = CT_edge_lambda
        self.nerf_lambda = nerf_lambda

    def pre_train_G(self, NORMAL_XRAY, prob_XRAY, real_CT, generator, optimizer_G):
        gen_CTs, gen_xrays, _ = generator(NORMAL_XRAY, prob_XRAY)

        xray_max = torch.amax(gen_xrays, dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        xray_min = torch.amin(gen_xrays, dim=(1, 2, 3)).reshape(-1, 1, 1, 1)
        norm_gen_xrays = (gen_xrays - xray_min) / (xray_max - xray_min)

        MSE2D = nn.L1Loss()(NORMAL_XRAY, norm_gen_xrays)
        MSE3D = nn.L1Loss()(real_CT, gen_CTs)

        grad_x_real, grad_y_real, grad_45_real, grad_135_real = Gradient_edge(real_CT)
        grad_x, grad_y, grad_45, grad_135 = Gradient_edge(gen_CTs)
        EDGE_MSE = ((nn.MSELoss()(grad_x_real, grad_x) + nn.MSELoss()(grad_y_real, grad_y) +
                     nn.MSELoss()(grad_45_real, grad_45) + nn.MSELoss()(grad_135_real, grad_135)))

        g_loss = MSE3D + MSE2D * self.nerf_lambda + EDGE_MSE * self.CT_edge_lambda

        optimizer_G.zero_grad()

        g_loss.backward()

        optimizer_G.step()

        torch.cuda.empty_cache()

        return g_loss.detach().item(), MSE2D.detach().item(), MSE3D.detach().item(), EDGE_MSE.detach().item()

    def snapshot(self, generator, input_dir, output_dir, MEAN_XRAY_PATH, STD_XRAY_PATH, step, device, photo_num=10):
        os.makedirs(os.path.join(output_dir, str(step)), exist_ok=True)

        device_type = device

        xray = np.load(input_dir).astype(np.float32)
        xray = (xray - np.min(xray)) / (np.max(xray) - np.min(xray))
        xray = torch.from_numpy(xray).to(device_type)

        MEAN_XRAY = np.load(MEAN_XRAY_PATH).astype(np.float32)
        MEAN_XRAY = torch.from_numpy(MEAN_XRAY).to(device_type)

        STD_XRAY = np.load(STD_XRAY_PATH).astype(np.float32)
        STD_XRAY = torch.from_numpy(STD_XRAY).to(device_type)

        prob_XRAY = negative_likelihood(xray, MEAN_XRAY, STD_XRAY)

        input_xray = torch.unsqueeze(xray, dim=0)
        prob_XRAY = torch.unsqueeze(prob_XRAY, dim=0)

        with torch.no_grad():
            gen_CTs, gen_xrays, _ = generator(input_xray, prob_XRAY)
            ct_image = gen_CTs[0, 0].cpu().numpy()
            drr_image = gen_xrays[0, 0].cpu().numpy()

        cv2.imwrite(os.path.join(output_dir, str(step), 'drr.png'), self.normalize(drr_image))

        for i in range(0, 128, photo_num):
            cv2.imwrite(os.path.join(output_dir, str(step), 'CT_{}.png'.format(i)), self.normalize(ct_image[i]))

    def normalize(self, photo):
        photo = (photo - np.min(photo)) * 255 / (np.max(photo) - np.min(photo))
        return photo.astype(np.uint8)
