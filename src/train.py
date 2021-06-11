#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:01:08 2021

@author: danish
"""

import torch
import os
from datagen import read_save_data, load_data, get_data_loaders
from utils import get_device, get_optimizers, get_criterions
from networks import Generator, Discriminator


def get_models(device):
    # create models
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)
    return generator_A2B, generator_B2A, discriminator_A, discriminator_B


def update_generators(generators, optimizer_G, real_imgs, identity_loss):
    #generator models
    generator_B2A, generator_A2B = generators
    #data
    real_img_A, real_img_B = real_imgs
    # Set gradients of generator_A and generator_B to zero
    optimizer_G.zero_grad()
    # Identity loss
    # G_B2A(A) should equal A if real A is fed
    id_img_A = generator_B2A(real_img_A)
    identity_loss_A = identity_loss(id_img_A, real_img_A) * 5.0
    # G_A2B(B) should equal B if real B is fed
    id_img_A = generator_A2B(real_img_B)
    identity_loss_B = identity_loss(id_img_A, real_img_B) * 5.0
    return identity_loss_A, identity_loss_B


def train_cycle_gan(data, epochs, batch_size, lr, img_size):
    device = get_device(True)
    PATCH_SHAPE = 16 
    #seprating data
    train_A, train_B = data
    # calculate the number of batches per training epoch
    num_batches = int(len(train_A)/batch_size)
    
    #get models
    models = get_models(device)
    generator_A2B, generator_B2A, discriminator_A, discriminator_B = models
    #get loss functions
    cycle_loss, identity_loss, adversarial_loss = get_criterions(device)
    #get optimizers
    optimizer_G, optimizer_D_A, optimizer_D_B = get_optimizers(models, lr)
    #get data loaders
    dataloader_A, dataloader_B = get_data_loaders(data, batch_size, img_size,
                                                  PATCH_SHAPE)
    
    
    for epoch in range(epochs):
        for i, (data_A, data_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_img_A, Y_A = data_A[0].to(device), data_A[1].to(device)
            real_img_B, Y_B = data_B[0].to(device), data_B[1].to(device)
            
            ################# Update Generators A2B and B2A #################
            #identity loss: {G_B2A(A), A} & {G_A2B(B), B}
            generators = (generator_B2A, generator_A2B)
            real_imgs = (real_img_A, real_img_B)
            identity_loss_A, identity_loss_B = update_generators(generators, 
                                                                 optimizer_G, 
                                                                 real_imgs, 
                                                                 identity_loss)
            
            
            
    
    
    
if __name__ == '__main__':
    IMG_SIZE = (256, 256)
    PATH = '../dataset/data'
    FILE_NAME = 'horse2zebra'
    DATA_PATH = os.path.join(PATH, FILE_NAME+'.npz')
    if os.path.exists(DATA_PATH):
        print(f'\nLoading data from path: {DATA_PATH}')
        data = load_data(DATA_PATH)
    else:
        read_path = os.path.join(PATH, FILE_NAME)
        print(f'Reading and saving data from path: {read_path}')
        data = read_save_data(read_path, IMG_SIZE)
    BATCH_SIZE = 1
    
        