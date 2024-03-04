import os
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from generator import renderers
from processes import process
from dataset import datasets
import configs as configs
from tqdm import tqdm
import matplotlib.pyplot as plt


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def set_generator(config, device, opt):
    generator = getattr(renderers, config['generator']['class'])(**config['generator']['kwargs'])

    if opt.load_dir != '':
        generator.load_state_dict(
            torch.load(os.path.join(opt.load_dir, 'step%06d_generator.pth' % opt.set_step), map_location='cpu'), strict=False)

    generator = generator.to(device)

    return generator


def set_optimizer_G(generator, config, opt):
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['optimizer']['gen_lr'],
                                   betas=config['optimizer']['betas'])
    if opt.load_dir != '':
        state_dict = torch.load(os.path.join(opt.load_dir, 'step%06d_optimizer_G.pth' % opt.set_step),
                                map_location='cpu')
        optimizer_G.load_state_dict(state_dict)

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    return optimizer_G, lr_scheduler_G


def training_process(opt, device):
    # --------------------------------------------------------------------------------------
    # extract training config
    config = getattr(configs, opt.config)

    # --------------------------------------------------------------------------------------
    # set generator

    generator = set_generator(config, device, opt)

    for name, param in generator.named_parameters():
        print(f'{name:<{96}}{param.shape}')
    total_num = sum(p.numel() for p in generator.parameters())
    print('G: Total ', total_num)

    # --------------------------------------------------------------------------------------
    # set optimizers
    optimizer_G, lr_scheduler_G = set_optimizer_G(generator, config, opt)

    torch.cuda.empty_cache()

    if opt.set_step != None:
        generator.step = opt.set_step

    # ----------
    #  Training
    # ----------
    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(opt.config))
        f.write('\n\n')
        f.write(str(config))

    total_progress_bar = tqdm(total=opt.n_epochs, desc="Total progress",
                              dynamic_ncols=True)  # Keeps track of total training
    total_progress_bar.update(generator.epoch)  # Keeps track of progress to next stage
    interior_step_bar = tqdm(desc="Steps", dynamic_ncols=True)

    # --------------------------------------------------------------------------------------
    # set loss
    each_process = getattr(process, config['process']['class'])(**config['process']['kwargs'])

    # --------------------------------------------------------------------------------------
    # get dataset
    dataset = getattr(datasets, config['dataset']['class'])(**config['dataset']['kwargs'])
    dataloader = DataLoader(dataset, batch_size=config['global']['batch_size'], shuffle=True, num_workers=4)

    # --------------------------------------------------------------------------------------
    # main training loop

    for _ in range(opt.n_epochs):
        total_progress_bar.update(1)

        sum_MSE2D_penalty = 0
        sum_MSE3D_penalty = 0
        sum_EDGE_penalty = 0

        # --------------------------------------------------------------------------------------
        # trainging iterations
        for i, (CT, NORMAL_XRAY, prob_XRAY) in enumerate(dataloader):

            # save model
            if generator.step % opt.model_save_interval == 0:
                torch.save(generator.state_dict(),
                           os.path.join(opt.output_dir, 'step%06d_generator.pth' % generator.step))
                torch.cuda.empty_cache()

            real_CT = CT.to(device)
            NORMAL_XRAY = NORMAL_XRAY.to(device)
            prob_XRAY = prob_XRAY.to(device)

            generator.train()

            # --------------------------------------------------------------------------------------
            # TRAIN GENERATOR
            g_loss, g_MSE2D, g_MSE3D, g_EDGE = each_process.pre_train_G(NORMAL_XRAY, prob_XRAY, real_CT,
                                                                        generator, optimizer_G)

            sum_MSE2D_penalty += g_MSE2D
            sum_MSE3D_penalty += g_MSE3D
            sum_EDGE_penalty += g_EDGE

            # --------------------------------------------------------------------------------------
            # print and save
            interior_step_bar.update(1)
            if i % 5 == 0:
                tqdm.write(
                    f"[lr: {optimizer_G.param_groups[0]['lr']}] "
                    f"[Step: {generator.step}] [G loss: {g_loss}] "
                    f"[Epoch: {generator.epoch}/{opt.n_epochs}] [Img Size: {config['global']['img_size']}] "
                    f"[Batch Size: {config['global']['batch_size']}]")


            # save fixed angle generated images
            if generator.step > 0 and generator.step % opt.sample_interval == 0:
                each_process.snapshot(generator, config['snapshot']['test_dir'], opt.output_dir,
                                      config['snapshot']['MEAN_XRAY_PATH'], config['snapshot']['STD_XRAY_PATH'],
                                      generator.step, device)

            # --------------------------------------------------------------------------------------
            generator.step += 1
        generator.epoch += 1

        lr_scheduler_G.step()

        with open(os.path.join(opt.output_dir, 'generator.csv'), 'a') as f:
            f.writelines(
                "{},{},{},{}\n".format(str(generator.epoch),
                                       sum_MSE2D_penalty / len(dataloader),
                                       sum_MSE3D_penalty / len(dataloader),
                                       sum_EDGE_penalty / len(dataloader)))
