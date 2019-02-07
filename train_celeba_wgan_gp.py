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
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate') 
parser.add_argument('--batch_size', default=24, type=int)
parser.add_argument('--resume', '-r', type=int, default=0, help='resume from checkpoint')
parser.add_argument('--epochs', '-e', type=int, default=50, help='number of args.epochs to train.')
parser.add_argument('--n_critic', default=5, type=int, help="traing generator in these many args.epochs as compared to critic")
parser.add_argument('--z_dim', default=100, type=int)

parser.add_argument('--alr', default=0.001, type=float, help="learning rate for classifier")
parser.add_argument('--aresume', '-ar', type=int, default=0, help='resume classifier from checkpoint')

args = parser.parse_args()
net = models.AsthecitiyClassifier()
D = models.DiscriminatorWGANGP(3)
G = models.Generator(args.z_dim)

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
    global net
    sepoch, step, best_acc = 0, 0, 0
    if(args.aresume):
        if(os.path.isfile('./save/aes/network.ckpt')):
            net.load_state_dict(torch.load('./save/aes/network.ckpt'))
        print('==> Network : loaded')

        if(os.path.isfile("./save/aes/info.txt")):
            with open("./save/aes/info.txt", "r") as f:
                sepoch, step = (int(i) for i in str(f.read()).split(" "))
            print("=> Network : prev epoch found")
    else :
        with open("./save/aes/logs/train_loss.log", "w+") as f:
            pass

    dataloader = iter(augument_data(args.batch_size))
    le = len(dataloader) - 1
    
    params = net.parameters()     
    optimizer = torch.optim.Adam(params, lr=args.alr) 
    criterion = torch.nn.BCELoss()
    
    for epoch in range(sepoch, args.epochs):
        train_loss, accu1 = 0.0, 0.0
        for _ in range(step, le):
            real_img, label1, fake_img, label0 = next(dataloader)
            imgs, labels = torch.cat((real_img, fake_img)), torch.cat((label1, label0)).type(torch.FloatTensor)
            imgs = imgs.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            predictions = net(imgs)
            loss = criterion(predictions, labels)
            loss.backward()
            
            tl = loss.item()
            train_loss += tl
            pred = torch.max(predictions, 1)[0].type(torch.LongTensor)
            accu = (pred == labels).sum().item()
            accu1 += accu

            gc.collect()
            torch.cuda.empty_cache()

            torch.save(net.state_dict(), './save/aes/network.ckpt')
            with open("./save/aes/info.txt", "w+") as f:
                f.write("{} {}".format(epoch, i))

            with open("./save/aes/logs/train_loss.log", "a+") as lfile:
                lfile.write("{}\n".format(tl))

            progress_bar(i, len(dataloader), 'Loss: {}, Accuracy: {} '.format(tl, accu))
        print('=> Network : Epoch [{}/{}], Loss:{:.4f}, Accuracy:{:.4f}'.format(epoch + 1, args.epochs, train_loss / le, accu1 / le))
        old_best = best_acc
        best_acc = max(best_acc, accu1/le)
        if(best_acc != old_best):
            torch.save(net.state_dict(), './save/aes/best.ckpt')
        print("Best Metrics : {}".format(best_acc))


def train_wgan(train_a=False):
    global G
    global D
    global net

    data_loader = gan_data(args.batch_size)
    utils.cuda([D, G])

    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion = torch.nn.CrossEntropyLoss()
    start_epoch = 0
    ckpt_dir = './checkpoints/celeba_wgan_gp'
    utils.mkdir(ckpt_dir)
    if(args.resume):
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
            if(train_a):
                a = net(f_imgs.detach())
                targets = torch.ones(a.shape[0]).type(torch.LongTensor).to(device)
                al = criterion(a, targets)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance 
            gp = gradient_penalty(imgs.data, f_imgs.data, D)
            d_loss = -wd + gp * 10.0 

            if(train_a):
                d_loss += al

            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            dl += d_loss.data.cpu().numpy()
            d_count += 1

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

                print("Epoch: ({}) ({}/{}) - D Loss : {:.4f} - G Loss : {:.4f}".format(epoch, i + 1, len(data_loader), d_loss.data.cpu().numpy(), g_loss.data.cpu().numpy()), end="\r")

            if (i + 1) % 100 == 0:
                G.eval()
                f_imgs_sample = (G(z_sample).data + 1) / 2.0

                save_dir = './sample_images_while_training/celeba_wgan_gp'
                utils.mkdir(save_dir)
                torchvision.utils.save_image(f_imgs_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, i + 1, len(data_loader)), nrow=10)


        print("Epoch {} - D Loss : {:.4f} - G Loss : {:.4f}".format(epoch, int(dl / d_count), int(gl / g_count)))
        utils.save_checkpoint({'epoch': epoch + 1, 'D': D.state_dict(), 'G': G.state_dict(), 'd_optimizer': d_optimizer.state_dict(), 'g_optimizer': g_optimizer.state_dict()}, '%s/Epoch_(%d).ckpt' % (ckpt_dir, epoch + 1), max_keep=2)


if __name__ == '__main__':
    train_aesthecity()
    train_wgan(train_a=True)
