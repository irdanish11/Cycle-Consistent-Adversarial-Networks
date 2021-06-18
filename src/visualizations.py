#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 19:48:31 2021

@author: danish
"""

from utils import combine_histories, read_pickle
import matplotlib.pyplot as plt
import os
import numpy as np



def plot(history, path, title):
    x_limit = len(list(history.values())[0])
    x = np.arange(1, x_limit+1)
    for k, v in history.items():
        plt.plot(x, v, label=k)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path, dpi=800)
    plt.show()


if __name__ == '__main__':
    hist_path = 'dump/horse2zebra'
    #combine_histories(hist_path)
    
    history = read_pickle(os.path.join(hist_path, 'history.pkl'))
    
    #generator loss
    generators_loss = {'Combined Generator Loss':history['gen_loss'], 
                       'Cycle Loss':history['gen_cycle_loss'],
                       'Adversarial Loss':history['gen_adv_loss'],
                       'Identity Loss':history['gen_id_loss']
                       }
    plot(generators_loss, path='./generator_losses.png', 
         title='Generator Losses')
    
    #discriminator loss
    generators_loss = {'Combined Discriminator Loss':history['disc_loss'], 
                       'Discriminator A Loss':history['disc_loss_A'],
                       'Adversarial Loss':history['disc_loss_B']
                       }
    plot(generators_loss, path='./discriminator_losses.png', 
         title='Discriminator Losses')
