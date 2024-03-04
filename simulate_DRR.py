import torch

from diffdrr.drr import DRR
from diffdrr.data import read_dicom

from pydicom import dcmread
from typing import Union
from pathlib import Path
import numpy as np
import cv2
import pydicom
import scipy.ndimage
import time
import os


def resample(image, spacing, new_spacing=[2.5, 2.5, 2.5]):
    # x, y, z
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)

    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image, new_spacing


def crop(image, shape=[128, 128, 128]):
    if not isinstance(spacing, np.ndarray):
        shape = np.array(shape)

    if image.shape[-1] > shape[-1]:
        mid = image.shape[-1] // 2
        mid_crop_image = image[..., mid - shape[-1] // 2: mid + shape[-1] - shape[-1] // 2]

    else:
        diff = shape[-1] - image.shape[-1]
        min_image = np.min(image)
        mid_crop_image = np.ones((image.shape[0], image.shape[1], shape[-1])) * min_image
        mid_crop_image[..., diff // 2: shape[-1] - diff + diff // 2] = image

    if image.shape[0] > shape[0]:
        H_mid = image.shape[0] // 2
        crop_image = mid_crop_image[H_mid - shape[0] // 2: H_mid + shape[0] - shape[0] // 2,
                     H_mid - shape[0] // 2: H_mid + shape[0] - shape[0] // 2]
    else:
        diff = shape[0] - image.shape[0]
        min_image = np.min(mid_crop_image)
        crop_image = np.ones((shape)) * min_image
        crop_image[diff // 2: shape[0] - diff + diff // 2, diff // 2: shape[0] - diff + diff // 2] = mid_crop_image

    return crop_image


# data path
LIDC_IDRI_PATH = '../DiffDRR/LIDC-IDRI/'
# CT save path
crop_file_PATH = './crop_image/'
os.makedirs(crop_file_PATH, exist_ok=True)

# X-ray save path
drr_PATH = './drr_Xray/'
os.makedirs(drr_PATH, exist_ok=True)

# parameters save path
Imaging_Parameters_PATH = './DRR_Parameters/'
os.makedirs(Imaging_Parameters_PATH, exist_ok=True)

i = 0
for each_file in sorted(os.listdir(LIDC_IDRI_PATH)):
    st = time.time()
    file_path = os.path.join(LIDC_IDRI_PATH, each_file)

    volume, spacing, dis, ratio = read_dicom(file_path)
    resample_image, new_spacing = resample(volume, spacing)
    crop_image = crop(resample_image)

    assert crop_image.shape[0] == 128 and crop_image.shape[1] == 128 and crop_image.shape[2] == 128, 'shape is wrong'

    np.save(crop_file_PATH + each_file + '.npy', crop_image)

    crop_image = (crop_image - np.min(crop_image)) / (np.max(crop_image) - np.min(crop_image))

    drr = DRR(
        crop_image,  # The CT volume as a numpy array
        np.array([2.5, 2.5, 2.5]),  # Voxel dimensions of the CT
        sdr=949 // 2,  # Source-to-detector radius (half of the source-to-detector distance)
        height=128,  # Height of the DRR (if width is not seperately provided, the generated image is square)
        delx=5,  # Pixel spacing (in mm)
        batch_size=1,  # How many batches of parameters will be passed = number of DRRs generated each forward pass
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    rotations = torch.tensor([[torch.pi, 0.0, torch.pi / 2]])
    translations = torch.tensor(crop_image.shape) * torch.tensor(new_spacing) / 2
    translations = translations.unsqueeze(0)

    # Generate the DRR
    drr.move_carm(rotations, translations)
    with torch.no_grad():
        img, alphas, idxs, source, target = drr()  # Only keep the graph if optimizing DRRs

    # (1, 128, 128)
    np.save(drr_PATH + each_file + '.npy', img[0].cpu().numpy())

    if i == 0:
        step_length = torch.diff(alphas, dim=-1)
        step_length = step_length.cpu().numpy()
        step_length = np.where(np.isnan(step_length), 0, step_length)
        np.save(Imaging_Parameters_PATH + 'step_length.npy', step_length)
        np.save(Imaging_Parameters_PATH + 'idxs.npy', idxs.cpu().numpy())
        np.save(Imaging_Parameters_PATH + 'source.npy', source.cpu().numpy())
        np.save(Imaging_Parameters_PATH + 'target.npy', target.cpu().numpy())

    print('processing{}/{}, using{}s'.format(i + 1, len(os.listdir(LIDC_IDRI_PATH)), time.time() - st))
    i += 1
