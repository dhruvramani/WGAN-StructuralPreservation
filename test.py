import os
import torch
import utils
import models
import numpy as np
from metrics import compute_score_raw

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_dir = './checkpoints/celeba_wgan_gp'


ckpt = utils.load_checkpoint(ckpt_dir)
G = models.Generator(100).to(device)
G.load_state_dict(ckpt['G'])

scores = compute_score_raw('celeba', 64, '/home/nevronas/dataset/img_align_celeba', sampleSize, 24, '../save/metrics/real', '../save/metrics/fake', G, 100, conv_model='resnet34', workers=4)
print(scores[30])