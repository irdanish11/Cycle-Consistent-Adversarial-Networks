#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:07:48 2021

@author: danish
"""

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data
import torch.nn.parallel
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.res_block = nn.Sequential(
            #first convolutional layer
            nn.Conv2d(in_channels=channels, out_channels=channels, 
                      kernel_size=3),
            nn.InstanceNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            #nn.ReflectionPad2d(1),
            #second convolutional layer
            nn.Conv2d(in_channels=channels, out_channels=channels, 
                      kernel_size=3),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        return x + self.res_block(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            #1st layer
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, 
                      stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #2nd layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, 
                      stride=2, padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),
            #3rd layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, 
                      stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #4th layer
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, 
                      padding=1),
            nn.InstanceNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            #5th layer
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, 
                      padding=1),
            #patch output
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, 
                      padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        return x
    
