import os
import torch
import Augmentor
import numpy as np
import torch.nn as nn
import PIL.Image as Image
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import utils

_DATASET_PATH = '/home/nevronas/dataset/img_align_celeba'

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser

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

        print(real_img, fake_img)

        return torch.stack((real_img, fake_img)), torch.Tensor([1, 0]).type(torch.LongTensor)

def augument_data(batch_size):
    celeba_dataset = AugumentedDataset()
    data_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader

if __name__ == '__main__':
    dataloader = augument_data(23)
    for x in enumerate(dataloader):
        print(x)
        break