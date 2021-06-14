#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:01:08 2021

@author: danish
"""


from datagen import read_save_data, load_data, get_data_loaders
from nn import get_models, get_criterions, get_optimizers
from utils import save_images, convert_seconds,save_models
from utils import get_device, write_pickle, print_inline
from utils import get_info_string, update_epoch_stats
from utils import load_checkpoints
from datagen import ImagePool
import argparse
import torch
import time
import os


#parser
parser = argparse.ArgumentParser(description='Cycle GAN using PyTorch.')
parser.add_argument("--dumppath", type=str, default="./dump",
                    help="path to directory where checkpoints will be stored.")
args = parser.parse_args()
print('Dump Path: ', args.dumppath)


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
                      loss_functions, epoch_history):
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
    #summation of respective losses
    id_loss = identity_loss_A + identity_loss_B
    adv_loss = adversarial_loss_B2A + adversarial_loss_A2B
    cy_loss = cycle_loss_ABA + cycle_loss_BAB
    #Combined loss
    combined_gan_loss = id_loss + adv_loss + cy_loss
    #history update
    epoch_history['gen_loss'].append(combined_gan_loss.item())
    epoch_history['gen_id_loss'].append(id_loss.item())
    epoch_history['gen_adv_loss'].append(adv_loss.item())
    epoch_history['gen_cycle_loss'].append(cy_loss.item())
    return combined_gan_loss, fake_imgs, epoch_history
    

def compute_discriminator_loss(discriminator, data, pool, loss_functions):
    adversarial_loss = loss_functions[2]
    #unpacking data
    real_img, real_label, fake_img, fake_label = data 
                               
    #Real image loss
    real_output = discriminator(real_img)
    disc_real_loss = adversarial_loss(real_output, real_label)

    #Fake image loss
    fake_img = pool.update_image_pool(fake_img)
    fake_output = discriminator(fake_img.detach())
    disc_fake_loss = adversarial_loss(fake_output, fake_label)

    #Combined loss
    loss_discriminator = (disc_real_loss + disc_fake_loss) / 2
    return loss_discriminator


def train_cycle_gan(data, epochs, batch_size, lr, img_size, dump_path):
    device = get_device(True)
    #seprating data
    train_A, train_B = data
    # calculate the number of batches per training epoch
    num_batches = int(len(train_A)/batch_size)
    
    #get models
    models = get_models(device)
    #checking if checkpoints present then resume training
    start_epoch, models = load_checkpoints(models, dump_path)
    #unpacking models
    generator_A2B, generator_B2A, discriminator_A, discriminator_B = models
    #get loss functions
    loss_functions = get_criterions(device)
    #get optimizers
    optimizer_G, optimizer_D_A, optimizer_D_B = get_optimizers(models, lr)
    #get data loaders
    dataloader_A, dataloader_B = get_data_loaders(data, batch_size, img_size)
    #image pools for discriminators
    pool_A = ImagePool(max_size=50)
    pool_B = ImagePool(max_size=50)
    
    history = {'gen_loss':[], 'disc_loss':[], 'gen_id_loss':[], 
               'gen_adv_loss':[], 'gen_cycle_loss':[], 'disc_loss_A':[],
               'disc_loss_B':[]}
    print('\n\t\t\t________________________________________________________\n')
    for epoch in range(start_epoch, epochs):
        start = time.time()
        epoch_history = {'gen_loss':[], 'disc_loss':[], 'gen_id_loss':[], 
                         'gen_adv_loss':[], 'gen_cycle_loss':[], 
                         'disc_loss_A':[], 'disc_loss_B':[]}
        for i, (data_A, data_B) in enumerate(zip(dataloader_A, dataloader_B)):
            real_img_A = data_A.to(device)
            real_img_B = data_B.to(device)
            
            # real data label is 1, fake data label is 0.
            real_label = torch.full((batch_size, 1), 1, device=device, 
                                    dtype=torch.float32)
            fake_label = torch.full((batch_size, 1), 0, device=device, 
                                    dtype=torch.float32)
            #coupling models and data
            generators = (generator_B2A, generator_A2B)
            discriminators = discriminator_A, discriminator_B
            real_imgs = (real_img_A, real_img_B)
            
            ##################################################################
            #                       Update Generators A2B and B2A
            ##################################################################
            # Set gradients of generator_A and generator_B to zero
            optimizer_G.zero_grad()
            #forward pass and loss computation
            combined_gan_loss, fake_imgs, epoch_history = forward_generators(
                generators, discriminators, real_imgs, real_label, 
                loss_functions, epoch_history)
            #Compute gradients for generator_A and generator_B
            combined_gan_loss.backward()
            #Update generator_A and generator_B's weights
            optimizer_G.step()
            
            
            ##################################################################
            #                       Update Discriminators A and B
            ##################################################################
            #unpacking fake data
            fake_img_A, fake_img_B = fake_imgs
            
            ##################### Updating Discriminator A ###################
            data_A = (real_img_A, real_label, fake_img_A, fake_label) 
            # Set discriminator_A gradients to zero
            optimizer_D_A.zero_grad()
            #forward pass and loss computation for Discriminator A
            loss_discriminator_A = compute_discriminator_loss(discriminator_A, 
                                                              data_A, pool_A, 
                                                              loss_functions)
            #Calculate gradients for discriminator_A
            loss_discriminator_A.backward()
            #Update discriminator_A weights
            optimizer_D_A.step()
            #history update
            epoch_history['disc_loss_A'].append(loss_discriminator_A.item())
            
            ##################### Updating Discriminator B ###################
            data_B = (real_img_B, real_label, fake_img_B, fake_label) 
            # Set discriminator_B gradients to zero
            optimizer_D_B.zero_grad()
            #forward pass and loss computation for Discriminator B
            loss_discriminator_B = compute_discriminator_loss(discriminator_B, 
                                                              data_B, pool_B, 
                                                              loss_functions)
            #Calculate gradients for discriminator_B
            loss_discriminator_B.backward()
            #Update discriminator_B weights
            optimizer_D_B.step()
            #history update
            epoch_history['disc_loss_B'].append(loss_discriminator_B.item())
            
            #combined loss of discriminators
            epoch_history['disc_loss'].append((loss_discriminator_A
                                               +loss_discriminator_B).item())
            #print info for each batch
            info = get_info_string(i, epoch_history)
            batch_info = f'Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{num_batches} | '
            print_inline(batch_info+info)
        time_taken = convert_seconds(int(time.time() - start))
        #compute statistics of one epoch
        update_epoch_stats(epoch, epoch_history, history, time_taken)
        #saving models
        save_models(models, dump_path, epoch+1)
        #saving images
        save_images(real_imgs, generators, dump_path, epoch+1)
        #saving history
        write_pickle(history, os.path.join(dump_path, 'history.pkl'))
        print('\t\t\t________________________________________________________\n')
        print('\n\n')
    return models, history
            

if __name__ == '__main__':
    IMG_SIZE = (256, 256)
    PATH = '../dataset/data'
    FILE_NAME = 'horse2zebra'
    DUMP_PATH = os.path.join(args.dumppath, FILE_NAME)
    DATA_PATH = os.path.join(PATH, FILE_NAME+'.npz')
    os.makedirs(DUMP_PATH, exist_ok=True)
    
    #loading or creating data
    if os.path.exists(DATA_PATH):
        print(f'\nLoading data from path: {DATA_PATH}')
        data = load_data(DATA_PATH)
    else:
        read_path = os.path.join(PATH, FILE_NAME)
        print(f'Reading and saving data from path: {read_path}')
        data = read_save_data(read_path, IMG_SIZE)
        print(f'\nLoading data from path: {DATA_PATH}')
        data = load_data(DATA_PATH)
    
    #parameters for training
    BATCH_SIZE = 1
    EPOCHS = 100
    LR = 0.001
    models, history = train_cycle_gan(data, EPOCHS, BATCH_SIZE, LR, IMG_SIZE, 
                                      DUMP_PATH)
    
        