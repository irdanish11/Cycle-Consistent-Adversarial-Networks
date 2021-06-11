#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:00:01 2021

@author: danish
"""

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
import torch.nn as nn
import numpy as np
import itertools

def get_device(CUDA=True):
    CUDA = CUDA and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if CUDA:
        print("CUDA version: {}\n".format(torch.version.cuda))
    seed = np.random.randint(1, 10000)
    print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if CUDA else "cpu")
    print('Device: ', device)
    return device 

def get_criterions(device):
    # define loss function (adversarial_loss) and optimizer
    cycle_loss = torch.nn.L1Loss().to(device)
    identity_loss = torch.nn.L1Loss().to(device)
    adversarial_loss = torch.nn.MSELoss().to(device)
    return cycle_loss, identity_loss, adversarial_loss

def get_optimizers(models, lr):
    generator_A2B, generator_B2A, discriminator_A, discriminator_B = models
    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(generator_A2B.parameters(), 
                                                   generator_B2A.parameters()),
                                                   betas=(0.5, 0.999),
                                                   lr=lr)
    optimizer_D_A = torch.optim.Adam(discriminator_A.parameters(), lr=lr, 
                                     betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(discriminator_B.parameters(), lr=lr, 
                                     betas=(0.5, 0.999))
    return optimizer_G, optimizer_D_A, optimizer_D_B