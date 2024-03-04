# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/api/06_utils.ipynb.

# %% ../notebooks/api/06_utils.ipynb 3
# from __future__ import annotations

import torch

from .detector import Detector

# %% auto 0
__all__ = ['reshape_subsampled_drr']

# %% ../notebooks/api/06_utils.ipynb 4
def reshape_subsampled_drr(
    img: torch.Tensor,
    detector: Detector,
    batch_size: int,
):
    n_points = detector.height * detector.width
    drr = torch.zeros(batch_size, n_points).to(img)
    drr[:, detector.subsamples[-1]] = img
    drr = drr.view(batch_size, 1, detector.height, detector.width)
    return drr
