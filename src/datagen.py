#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:12:34 2021

@author: danish
"""

import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
import random

def read_save_data(path, size):
    dir_path = os.listdir(path)  
    data = {'A':[], 'B':[]}
    for d in dir_path:
        category = os.listdir(os.path.join(path, d))    
        for c in category:
            imgs_path = os.path.join(path, d, c)
            imgs_lst = glob(os.path.join(imgs_path, '*.jpg'))
            for img_path in tqdm(imgs_lst):
                img = cv2.imread(img_path)
                img = np.array(cv2.resize(img, size))
                data[c].append(img)
    fname = path+'.npz'
    # size_A, size_B = len(data['A']), len(data['B'])
    # min_size = size_A if size_A < size_B else size_B
    # a = np.array(data['A'][0:min_size])
    # b = np.array(data['B'][0:min_size])
    np.savez_compressed(fname, np.array(data['A']), np.array(data['B']))
    return data
    

def load_data(fname):
    data = np.load(fname)
    train_A, train_B = data['arr_0'], data['arr_1']
    return train_A, train_B
    

class ImageLoader(torch.utils.data.Dataset):
  def __init__(self, train_data, n_samples, transform):
        'Initialization'
        self.train_data = train_data
        self.n_samples = n_samples
        self.transform = transform
        self.total_samples = len(train_data)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.train_data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # choose random instances
        idx = np.random.randint(0, self.total_samples, self.n_samples)      
    	# retrieve selected images
        X = self.train_data[idx][0]
        X = self.transform(X)
        return X
    

class ImagePool:
    """
        Image pool of 50 generated images for each discriminator model that is 
        first populated and probabilistically either adds new images to the 
        pool by replacing an existing image or uses a generated image directly.
    """
    def __init__(self, max_size=50):
        assert (max_size > 0)
        self.max_size = max_size
        self.data = []

    def update_image_pool(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)
    
def get_data_loaders(data, batch_size, image_size):
    train_A, train_B = data
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.RandomCrop(image_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize((0.5, 0.5, 0.5), 
                                                         (0.5, 0.5, 0.5))])
    dataset_A = ImageLoader(train_A, n_samples=batch_size, transform=transform)
    dataset_B = ImageLoader(train_B, n_samples=batch_size, transform=transform)
    #train_data, n_samples, patch_shape, transform
    dataloader_A = torch.utils.data.DataLoader(dataset_A, batch_size)
    dataloader_B = torch.utils.data.DataLoader(dataset_B, batch_size)
    return dataloader_A, dataloader_B

if __name__ == '__main__':        
    data = read_save_data(path='../dataset/data/horse2zebra', size=(256, 256))
