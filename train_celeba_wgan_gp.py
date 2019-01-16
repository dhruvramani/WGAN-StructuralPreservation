from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import models
import argparse
import PIL.Image as Image
import tensorboardX
import torch
from torch.autograd import grad
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils

from dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" gpu """
gpu_id = [2]
utils.cuda_devices(gpu_id)

parser = argparse.ArgumentParser(description='PyTorch Facial Structural Preservation WGAN')
parser.add_argument('--args', default=0.0002, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=50, help='number of args.epochs to train.')
parser.add_argument('--n_critic', default=5, type=int, help="traing generator in these many args.epochs as compared to critic")
parser.add_argument('--z_dim', default=100, type=int)

args = parser.parse_args()

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.size(0)] + [1] * (x.dim() - 1)
    alpha = utils.cuda(torch.rand(shape))
    z = x + alpha * (y - x)

    # gradient penalty
    z = utils.cuda(Variable(z, requires_grad=True))
    o = f(z)
    g = grad(o, z, grad_outputs=utils.cuda(torch.ones(o.size())), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

    return gp

def train_aesthecity():


def train_wgan():
    """ data """
    data_loader = gan_data(args.batch_size)

    """ model """
    D = models.DiscriminatorWGANGP(3)
    G = models.Generator(args.z_dim)
    utils.cuda([D, G])

    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    """ load checkpoint """
    if(args.resume):
        ckpt_dir = './checkpoints/celeba_wgan_gp'
        utils.mkdir(ckpt_dir)
        try:
            ckpt = utils.load_checkpoint(ckpt_dir)
            start_epoch = ckpt['epoch']
            D.load_state_dict(ckpt['D'])
            G.load_state_dict(ckpt['G'])
            d_optimizer.load_state_dict(ckpt['d_optimizer'])
            g_optimizer.load_state_dict(ckpt['g_optimizer'])
            print("=> GAN : Loaded from checkpoint")
        except:
            print(' [*] No checkpoint!')
            start_epoch = 0


    """ run """
    writer = tensorboardX.SummaryWriter('./summaries/celeba_wgan_gp')

    z_sample = Variable(torch.randn(100, args.z_dim))
    z_sample = utils.cuda(z_sample)
    for epoch in range(start_epoch, args.epochs):
        d_count, g_count = 0, 0
        dl, gl = 0.0, 0.0
        for i, (imgs, _) in enumerate(data_loader):
            # step
            step = epoch * len(data_loader) + i + 1

            # set train
            G.train()

            # leafs
            imgs = Variable(imgs)
            bs = imgs.size(0)
            z = Variable(torch.randn(bs, args.z_dim))
            imgs, z = utils.cuda([imgs, z])

            f_imgs = G(z)

            # train D
            r_logit = D(imgs)            # f(x)
            f_logit = D(f_imgs.detach()) # f(G(z))

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance 
            gp = gradient_penalty(imgs.data, f_imgs.data, D)
            d_loss = -wd + gp * 10.0

            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            dl += d_loss.data.cpu().numpy()

            writer.add_scalar('D/wd', wd.data.cpu().numpy(), global_step=step)
            writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

            if step % args.n_critic == 0:
                # train G
                z = utils.cuda(Variable(torch.randn(bs, args.z_dim)))
                f_imgs = G(z)
                f_logit = D(f_imgs)
                g_loss = -f_logit.mean()

                D.zero_grad()
                G.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                gl += g_loss.data.cpu().numpy()
                g_count += 1
                writer.add_scalars('G',
                                   {"g_loss": g_loss.data.cpu().numpy()},
                                   global_step=step)

                print("Epoch: (%3d) (%5d/%5d) - D Loss : (%d) - G Loss : (%d)" % (epoch, i + 1, len(data_loader), d_loss.data.cpu().numpy(), g_loss.data.cpu().numpy()), end="\r")

            if (i + 1) % 100 == 0:
                G.eval()
                f_imgs_sample = (G(z_sample).data + 1) / 2.0

                save_dir = './sample_images_while_training/celeba_wgan_gp'
                utils.mkdir(save_dir)
                torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)


        print("Epoch {} - D Loss : {} - G Loss : {}".format(epoch, int(dl / d_count), int(gl / g_count)))
        utils.save_checkpoint({'epoch': epoch + 1, 'D': D.state_dict(), 'G': G.state_dict(), 'd_optimizer': d_optimizer.state_dict(), 'g_optimizer': g_optimizer.state_dict()}, '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1), max_keep=2)
