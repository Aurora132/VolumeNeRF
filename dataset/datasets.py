import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def negative_likelihood(image, mean, std):
    return torch.log(std) + (image - mean) ** 2 / (2 * (std ** 2))


class X2CT(Dataset):
    def __init__(self, CT_PATH, XRAY_PATH, MEAN_XRAY_PATH, STD_XRAY_PATH):
        super().__init__()
        self.CT_LIST = [os.path.join(CT_PATH, i) for i in sorted(os.listdir(CT_PATH))]
        self.XRAY_LIST = [os.path.join(XRAY_PATH, i) for i in sorted(os.listdir(XRAY_PATH))]
        self.MEAN_XRAY = np.load(MEAN_XRAY_PATH).astype(np.float32)
        self.STD_XRAY = np.load(STD_XRAY_PATH).astype(np.float32)

    def __len__(self):
        return len(self.CT_LIST)

    def __getitem__(self, index):
        CT = np.load(self.CT_LIST[index]).astype(np.float32)
        CT = (CT - np.min(CT)) / (np.max(CT) - np.min(CT))
        CT = torch.from_numpy(CT).permute(2, 0, 1)
        # C D H W
        CT = torch.unsqueeze(CT, dim=0)

        XRAY = np.load(self.XRAY_LIST[index]).astype(np.float32)
        NORMAL_XRAY = (XRAY - np.min(XRAY)) / (np.max(XRAY) - np.min(XRAY))
        NORMAL_XRAY = torch.from_numpy(NORMAL_XRAY)

        MEAN_XRAY = torch.from_numpy(self.MEAN_XRAY)
        STD_XRAY = torch.from_numpy(self.STD_XRAY)

        prob_XRAY = negative_likelihood(NORMAL_XRAY, MEAN_XRAY, STD_XRAY)
        return CT, NORMAL_XRAY, prob_XRAY


if __name__ == '__main__':
    CT_PATH = '../crop_image/train'
    XRAY_PATH = '../drr_Xray/train'
    MEAN_XRAY_PATH = '../DRR_Parameters/mean_xray.npy'
    STD_XRAY_PATH = '../DRR_Parameters/std_xray.npy'
    dataset = X2CT(CT_PATH, XRAY_PATH, MEAN_XRAY_PATH, STD_XRAY_PATH)
    CT, XRAY, prob_XRAY = dataset[50]
    print(CT.shape, torch.max(CT), torch.min(CT))
    print(XRAY.shape, torch.max(XRAY), torch.min(XRAY))
    print(prob_XRAY.shape, torch.max(prob_XRAY), torch.min(prob_XRAY))
    plt.subplot(141)
    plt.imshow(CT[0, 0].numpy(), cmap='gray')
    plt.subplot(142)
    plt.imshow(CT[0, -1].numpy(), cmap='gray')
    plt.subplot(143)
    plt.imshow(XRAY[0].numpy(), cmap='gray')

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for CT, NORMAL_XRAY, prob_XRAY in dataloader:
        print(CT.shape, NORMAL_XRAY.shape, prob_XRAY.shape)
        print(CT.requires_grad)
        break
    print(len(dataloader))
