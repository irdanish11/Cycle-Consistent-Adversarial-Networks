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
from datagen import ImagePool

def get_models(device):
    # create models
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    discriminator_A = Discriminator().to(device)
    discriminator_B = Discriminator().to(device)
    return generator_A2B, generator_B2A, discriminator_A, discriminator_B


def compute_identity_loss(generators, real_imgs, identity_loss):
    #generator models
    generator_B2A, generator_A2B = generators
    #data
    real_img_A, real_img_B = real_imgs
    # Identity loss
    # G_B2A(A) should equal A if real A is fed
    id_img_A = generator_B2A(real_img_A)
    identity_loss_A = identity_loss(id_img_A, real_img_A) * 5.0
    # G_A2B(B) should equal B if real B is fed
    id_img_A = generator_A2B(real_img_B)
    identity_loss_B = identity_loss(id_img_A, real_img_B) * 5.0
    return identity_loss_A, identity_loss_B

def compute_adversarial_loss(generators, discriminators, real_imgs, real_label, 
                             adversarial_loss):
    #generator models
    generator_B2A, generator_A2B = generators
    #discriminator models
    discriminator_A, discriminator_B = discriminators
    #data
    real_img_A, real_img_B = real_imgs
    fake_img_A = generator_B2A(real_img_B)
    fake_output_A = discriminator_A(fake_img_A)
    adversarial_loss_B2A = adversarial_loss(fake_output_A, real_label)
    # GAN loss D_B(G_B(B))
    fake_img_B = generator_A2B(real_img_A)
    fake_output_B = discriminator_B(fake_img_B)
    adversarial_loss_A2B = adversarial_loss(fake_output_B, real_label)
    adv_losses = (adversarial_loss_B2A, adversarial_loss_A2B)
    fake_imgs = (fake_img_A, fake_img_B)
    return adv_losses, fake_imgs

def compute_cycle_loss(generators, real_imgs, fake_imgs, cycle_loss):
    #generator models
    generator_B2A, generator_A2B = generators
    #real data
    real_img_A, real_img_B = real_imgs
    #fake data
    fake_img_A, fake_img_B = fake_imgs
    recovered_img_A = generator_B2A(fake_img_B)
    cycle_loss_ABA = cycle_loss(recovered_img_A, real_img_A) * 10.0

    recovered_img_B = generator_A2B(fake_img_A)
    cycle_loss_BAB = cycle_loss(recovered_img_B, real_img_B) * 10.0
    return cycle_loss_ABA, cycle_loss_BAB
    

def forward_generators(generators, discriminators, real_imgs, real_label, 
                      loss_functions):
    cycle_loss, identity_loss, adversarial_loss = loss_functions
    #Identity loss: {G_B2A(A), A} & {G_A2B(B), B}
    identity_loss_A, identity_loss_B = compute_identity_loss(generators, 
                                                        real_imgs, 
                                                        identity_loss)
    
    #GAN/Adversarial loss: D_A(G_A(A))
    adv_losses, fake_imgs = compute_adversarial_loss(generators, 
                                                     discriminators, 
                                                     real_imgs, 
                                                     real_label, 
                                                     adversarial_loss)
    adversarial_loss_B2A, adversarial_loss_A2B = adv_losses

    #Cycle loss: {G_B2A(G_A2B(A)), A}
    cycle_loss_ABA, cycle_loss_BAB = compute_cycle_loss(generators, 
                                                        real_imgs, 
                                                        fake_imgs, 
                                                        cycle_loss)

    #Combined loss
    combined_gan_loss = (identity_loss_A + identity_loss_B 
                         + adversarial_loss_B2A + adversarial_loss_A2B
                         + cycle_loss_ABA + cycle_loss_BAB)
    return combined_gan_loss, fake_imgs
    

def compute_discriminator_loss():
    #fake data
    fake_img_A, fake_img_B = fake_imgs
    
    
    adversarial_loss = loss_functions[2]
    
    # Real A image loss
    real_output_A = discriminator_A(real_img_A)
    disc_real_loss_A = adversarial_loss(real_output_A, real_label)

    # Fake A image loss
    fake_img_A = pool_A.update_image_pool(fake_img_A)
    fake_output_A = discriminator_A(fake_img_A.detach())
    disc_fake_loss_A = adversarial_loss(fake_output_A, fake_label)

    # Combined loss and calculate gradients
    loss_discriminator_A = (disc_real_loss_A + disc_fake_loss_A) / 2

    


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
    loss_functions = get_criterions(device)
    #get optimizers
    optimizer_G, optimizer_D_A, optimizer_D_B = get_optimizers(models, lr)
    #get data loaders
    dataloader_A, dataloader_B = get_data_loaders(data, batch_size, img_size,
                                                  PATCH_SHAPE)
    #image pools for discriminators
    pool_A = ImagePool(max_size=50)
    pool_B = ImagePool(max_size=50)
    
    for epoch in range(epochs):
        for i, (data_A, data_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_img_A, Y_A = data_A[0].to(device), data_A[1].to(device)
            real_img_B, Y_B = data_B[0].to(device), data_B[1].to(device)
            
            # real data label is 1, fake data label is 0.
            real_label = torch.full((batch_size, 1), 1, device=device, 
                                    dtype=torch.float32)
            fake_label = torch.full((batch_size, 1), 0, device=device, 
                                    dtype=torch.float32)
            #coupling models and data
            generators = (generator_B2A, generator_A2B)
            discriminators = discriminator_A, discriminator_B
            real_imgs = (real_img_A, real_img_B)
            
            ################# Update Generators A2B and B2A #################
            # Set gradients of generator_A and generator_B to zero
            optimizer_G.zero_grad()
            #forward pass and loss computation
            combined_gan_loss, fake_imgs = forward_generators(generators, 
                                                              discriminators, 
                                                              real_imgs, 
                                                              real_label, 
                                                              loss_functions)
            #Compute gradients for generator_A and generator_B
            combined_gan_loss.backward()
            #Update generator_A and generator_B's weights
            optimizer_G.step()
            
            ################# Update Discriminators A and B ##################
            # Set D_A gradients to zero
            optimizer_D_A.zero_grad()
            
            # Calculate gradients for D_A
            loss_discriminator_A.backward()
            # Update D_A weights
            optimizer_D_A.step()
            
            
            
            
    
    
    
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
    
        