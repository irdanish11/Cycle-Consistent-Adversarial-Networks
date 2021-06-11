#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:07:48 2021

@author: danish
"""

import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.res_block = nn.Sequential(
            #first convolutional layer
            nn.Conv2d(in_channels=channels, out_channels=channels, 
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            #nn.ReflectionPad2d(1),
            #second convolutional layer
            nn.Conv2d(in_channels=channels, out_channels=channels, 
                      kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_features=channels)
        )

    def forward(self, x):
        return x + self.res_block(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            ############## Encoder ##############
            # Initial convolution block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, 
                      padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),  
            # Downsampling
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, 
                      stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, 
                      stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            ############## Transformer ##############
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),
            ResNetBlock(channels=256),

            ############## Decoder ##############
            nn.ConvTranspose2d(in_channels=256, out_channels=128, 
                               kernel_size=3, stride=2, padding=0, 
                               output_padding=1),
            nn.InstanceNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, 
                               kernel_size=3, stride=2, padding=0, 
                               output_padding=1),
            nn.InstanceNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            # Output layer
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


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
    
