import argparse
import os
import time
import numpy as np

import torch
from datetime import datetime
from training_loop import training_process


def train(opt):
    # torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    training_process(opt, device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling(snapshot)")
    parser.add_argument('--output_dir', type=str, default='/home4/results/X2CT')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--config', type=str, default='X2CT')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=800)
    parser.add_argument("--DATE_FORMAT", type=str, default='%A_%d_%B_%Y_%Hh_%Mm_%Ss', help='time format')

    opt = parser.parse_args()

    systime = datetime.now().strftime(opt.DATE_FORMAT)

    opt.output_dir = os.path.join(opt.output_dir, systime)
    os.makedirs(opt.output_dir, exist_ok=True)

    print(opt)

    train(opt)
