# VolumeNeRF

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions for Use](#instructions-for-use)
- [License](./LICENSE)
- [Citation](#citation)

# 1. Overview

This project provides a Nerf-based CT volume reconstruction method using a projection single view. Using the code requires users to have basic knowledge about python, PyTorch, and deep neural networks.

# 2. Repo Contents
- [simulate_DRR.py](./simulate_DRR.py): DRR X-ray images generation.
- [generator/renderers.py](./generator/renderers.py): network definition of proposed method.
- [loss/edge_loss.py](./loss/edge_loss.py): edge loss functions definition.
- [dataset/datasets.py](./dataset/datasets.py): data loader definition for model training.
- [train.py](./train.py): main code for the training stage.
- [configs.py](./configs.py): main settings for the method.
 - You can perform your CT reconstruction by changing the settings here.


# 3. System Requirements

## Prerequisites
- Ubuntu 18.04
- NVIDIA GPU + CUDA (Geforce RTX 3090 with 24GB memory, CUDA 11.4)

## Package Versions
- python 3.8
- pytorch 1.10.1
- torchvision 0.11.2
- opencv-python 4.6.0.66
- numpy 1.19.5

# 4. Instructions for Use

## Training
- Run `python train_pre.py` to perform the first training process with the default setting. Then change the parameter `pretrain_dir` and run `python train_sec.py` to begin the second training process.
- Run `python train_SECT_DECT.py` to start the DECT imaging generation module training stage.

## Test demo
Run `python decompose_demo.py` to test the trained model deposited in `./checkpoints/decompose_four/` on the data in `./data/demo_data/`. Image results are stored in `./decompose_result/`.

# 5. License
This project is covered under the BSD-3-Clause License.
