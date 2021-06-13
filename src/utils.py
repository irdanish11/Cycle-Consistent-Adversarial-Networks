#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 18:00:01 2021

@author: danish
"""

import pickle
import sys
import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.parallel
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm


def get_device(cuda=True):
    cuda = cuda and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if cuda:
        print("CUDA version: {}\n".format(torch.version.cuda))
    seed = np.random.randint(1, 10000)
    print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if cuda else "cpu")
    print('Device: ', device)
    return device 


def write_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
   
        
def read_pickle(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_network(net, path):
    torch.save(net.state_dict(), path)
    
    
def load_network(path, net):
    net.load_state_dict(torch.load(path))
    return net


def get_info_string(idx, history):
    info = ('Generator Loss: {0:.4f} | '.format(history['gen_loss'][idx])
            + 'Discriminator Loss: {0:.4f} | '.format(history['disc_loss'][idx])
            + 'Identity Loss: {0:.4f} | '.format(history['gen_id_loss'][idx])
            + 'Adversarial Loss: {0:.4f} | '.format(history['gen_adv_loss'][idx])
            + 'Cycle Loss: {0:.4f} |'.format(history['gen_cycle_loss'][idx])
            + 'Discriminator Loss A: {0:.4f} | '.format(
                history['disc_loss_A'][idx])
            + 'Discriminator Loss B: {0:.4f} | '.format(
                history['disc_loss_B'][idx])
            )
    return info


def update_epoch_stats(idx, epoch_history, history, time_taken):
    #updating history
    history['gen_loss'].append(np.mean(epoch_history['gen_loss']))
    history['disc_loss'].append(np.mean(epoch_history['disc_loss']))
    history['gen_id_loss'].append(np.mean(epoch_history['gen_id_loss']))
    history['gen_adv_loss'].append(np.mean(epoch_history['gen_adv_loss']))
    history['gen_cycle_loss'].append(np.mean(epoch_history['gen_cycle_loss']))
    history['disc_loss_A'].append(np.mean(epoch_history['disc_loss_A']))
    history['disc_loss_B'].append(np.mean(epoch_history['disc_loss_B']))
    #time str
    time_str = f' - Time Taken by epoch: {time_taken}.\n'
    #getting complete epoch info
    info = get_info_string(idx, history)
    print('\nEpoch Completed.' + time_str + info)
    return history    

def print_inline(string):
    sys.stdout.write('\r'+string)
    sys.stdout.flush() 
    
    
def save_models(models, path, idx):
    ckpt_path = os.path.join(path, 'ckpt', str(idx))
    os.makedirs(ckpt_path, exist_ok=True)
    print('\nSaving Models\n')
    for model in tqdm(models):
        #save_network()
        model_path = os.path.join(ckpt_path, model.name+'.pth')
        save_network(model, model_path)
        
        
def save_images(real_imgs, generators, path, idx):
    img_path = os.path.join(path, 'images', str(idx))
    os.makedirs(img_path, exist_ok=True)
    #decoupling images
    real_img_A, real_img_B = real_imgs
    #decoupling generators
    #generator models
    generator_B2A, generator_A2B = generators
    #saving real images
    vutils.save_image(real_img_A, os.path.join(img_path, 'real_img_A.png'),
                      normalize=True)
    vutils.save_image(real_img_B, os.path.join(img_path, 'real_img_B.png'),
                      normalize=True)

    fake_img_A = 0.5 * (generator_B2A(real_img_B).detach() + 1.0)
    fake_img_B = 0.5 * (generator_A2B(real_img_A).detach() + 1.0)

    vutils.save_image(fake_img_A.detach(), 
                      os.path.join(img_path, 'fake_img_A.png'),
                      normalize=True)
    vutils.save_image(fake_img_B.detach(),
                      os.path.join(img_path, 'fake_img_B.png'),
                      normalize=True)    
    print(f'Images Saved at path: {img_path}') 
    

def convert_seconds(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60   
    return "%d:%02d:%02d" % (hour, minutes, seconds)
