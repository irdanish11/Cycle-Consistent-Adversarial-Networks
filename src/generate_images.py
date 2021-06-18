#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 00:31:12 2021

@author: danish
"""

import os
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
import cv2
from utils import get_device, load_serving_models
from PIL import Image
from datagen import get_test_loaders


def read_image(img_path):
    img = Image.open(img_path)
    img.resize((256, 256))
    return img

def tranlsate_images(img_A, image_size=(256, 256)):
    device = get_device(cuda=True)
    generator_A2B, generator_B2A = load_serving_models(ckpt_path, device)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(image_size),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), 
                                                         std=(0.5, 0.5, 0.5))])
    trans_img_A = torch.unsqueeze(transform(img_A), 0).to(device)
    fake_image_B = 0.5 * (generator_A2B(trans_img_A).data + 1.0)
    fake_image_B = fake_image_B.detach().cpu().numpy()[0]*255
    fake_image_B = np.transpose(fake_image_B.astype(np.uint8))
    cv2.imwrite('test2.jpg', fake_image_B)


def gen_directories(save_path):
    path_a = os.path.join(save_path, 'A')
    path_b = os.path.join(save_path, 'B')
    os.makedirs(path_a, exist_ok=True)
    os.makedirs(path_b, exist_ok=True)
    return path_a, path_b

def translate_test_imgs(test_path, image_size, save_path):
    path_a, path_b = gen_directories(save_path)
    #getting test loaders
    testloader_A, testloader_B = get_test_loaders(test_path, batch_size=1, 
                                                  image_size=image_size)
    #device to do inference on
    device = get_device(cuda=True)
    #loading weights
    generator_A2B, generator_B2A = load_serving_models(ckpt_path, device)
    #testing
    test_loop = tqdm(enumerate(zip(testloader_A, testloader_B)))
    for i, (data_A, data_B) in test_loop:
        real_img_A = data_A.to(device)
        real_img_B = data_B.to(device)
        #saving real images
        vutils.save_image(real_img_A, 
                          os.path.join(path_a, f'A{i}_real_img.png'),
                          normalize=True)
        vutils.save_image(real_img_B, 
                          os.path.join(path_b, f'B{i}_real_img.png'),
                          normalize=True)
        #computing images
        with torch.no_grad():
            fake_img_A = 0.5 * (generator_B2A(real_img_B).detach() + 1.0)
            fake_img_B = 0.5 * (generator_A2B(real_img_A).detach() + 1.0)
    
            vutils.save_image(fake_img_A.detach(), 
                              os.path.join(path_b, f'B{i}_fake_img.png'),
                              normalize=True)
            vutils.save_image(fake_img_B.detach(),
                              os.path.join(path_a, f'A{i}_fake_img.png'),
                              normalize=True)  
    print('\nAll images translated successfully!')




if __name__ == '__main__':
    #paths
    ckpt_path = 'dump/horse2zebra/ckpt/100'
    test_path = 'data'
    save_path = 'data/generated'
    
    
    #translate images
    image_size = (256, 256)
    translate_test_imgs(test_path, image_size, save_path)
    
    #load_imgs
    # img_path_A = 'test_data/A/n02381460_7500.jpg'
    # img_A = read_image(img_path_A)
    # tranlsate_images(img_A)
    # img_A.show()
    # img_A.save("test.jpg")
