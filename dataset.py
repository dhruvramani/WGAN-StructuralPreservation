import os
import torch
import Augmentor
import numpy as np
import torch.nn as nn
from skimage import io
import PIL.Image as Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
import datsetfolder as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import utils

_DATASET_PATH = '/home/nevronas/dataset/img_align_celeba'

def get_random_eraser(p=0.5, area_ratio_range=[0.02, 0.4], min_aspect_ratio=0.3, max_attempt=20):
    sl, sh = area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    def _random_erase(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(max_attempt):
            mask_area = np.random.uniform(sl, sh) * image_area
            aspect_ratio = np.random.uniform(rl, rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image
    return _random_erase

def gan_data(batch_size):
    crop_size = 108
    re_size = 64
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(crop),
         transforms.ToPILImage(),
         transforms.Resize(size=(re_size, re_size), interpolation=Image.BICUBIC),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

    imagenet_data = dsets.ImageFolder(_DATASET_PATH, transform=transform)
    data_loader = DataLoader(imagenet_data, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader

def skewed_transform(image):
    afine_tf = tf.AffineTransform(shear=0.2)
    modified = tf.warp(image, inverse_map=afine_tf)
    return modified

class AugumentedDataset(Dataset):
    def __init__(self, path=_DATASET_PATH):
        self.path = path
        self.earser = get_random_eraser()
        self.real_data = dsets.ImageFolder(self.path)

    def __len__(self):
        return int(self.real_data.__len__())

    def __getitem__(self, idx):
        real_img = self.real_data.__getitem__(idx)[0]
        real_np = np.array(real_img)
        fake_img = torch.Tensor(self.earser(real_np))
        real_np = np.array(real_img)
        fake_img2 = torch.Tensor(skewed_transform(real_np))
        real_labels, fake_labels = torch.tensor([1, 0]).type(torch.LongTensor), torch.tensor([0, 1]).type(torch.LongTensor)
        return real_img, real_labels, fake_img, fake_img2, fake_labels #torch.cat((real_img, fake_img)), torch.Tensor([1, 0]).type(torch.LongTensor)

def augument_data(batch_size):
    celeba_dataset = AugumentedDataset()
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True)
    return data_loader

if __name__ == '__main__':
    
    data_loader = iter(augument_data(23))
    real_img, label1, fake_img, fake_img2, label0 = next(data_loader)
    #img = np.array(mpimg.imread('/home/nevronas/dataset/img_align_celeba/1/011000.jpg'))
    #eraser = get_random_eraser()
    #fake_img = eraser(img)
    print(fake_img.shape, fake_img2.shape)
    mpimg.imsave("./out.png", real_img[0])
    mpimg.imsave("./out1.png", fake_img[0])
    mpimg.imsave("./out2.png", fake_img2[0])
    '''
    img = Image.fromarray(img, 'RGB')
    img.save('./foo.png')
    img.show() '''
