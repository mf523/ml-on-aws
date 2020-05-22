import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import os

import torch.optim as optim

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

class Generator(nn.Module):
    nz = None
    num_classes = None
    
    def __init__(self, *, nz, nc, ngf, num_classes):
        super(Generator, self).__init__()
        
        self.nz = nz
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        # Input is the latent vector Z + label.
        self.tconv1 = nn.ConvTranspose2d(self.nz + self.num_classes, ngf*8,
            kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf*8, ngf*4,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf*4, ngf*2,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf*2, ngf,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64


    def forward(self, z, labels):
        c = self.label_emb(labels)
        c = c.repeat(1, z.size(2), z.size(3)).view(c.size(0), c.size(1), z.size(2), z.size(3))
        x = torch.cat([z, c], 1)
        
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))

        return x


    def save(self, path, *, filename=None, device='cpu'):
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        self.to(device)
        if not filename is None:
            path = os.path.join(path, filename)
        torch.save(self.state_dict(), path)


    def load(self, path, *, filename=None):
        if not filename is None:
            path = os.path.join(path, filename)
        with open(path, 'rb') as f:
            self.load_state_dict(torch.load(f))

    
class Discriminator(nn.Module):
    nc = None
    ndf = None
    num_classes = None
    
    def __init__(self, *, nc, ndf, num_classes):
        super(Discriminator, self).__init__()
        
        self.nc = nc
        self.ndf = ndf
        self.num_classes = num_classes
        
        self.label_emb = nn.Embedding(self.num_classes, self.num_classes)

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(self.nc, self.ndf,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(self.ndf, self.ndf*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ndf*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(self.ndf*2, self.ndf*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ndf*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(self.ndf*4, self.ndf*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.ndf*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(self.ndf*8, 1, 4, 1, 0, bias=False)
        

    def forward(self, x, labels):
        c = self.label_emb(labels)
        c = c.repeat(1, x.size(1), x.size(2)).view(c.size(0), x.size(1), x.size(2), c.size(1))
        x = torch.cat([x, c], 3)
        
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = torch.sigmoid(self.conv5(x))

        return x


    def save(self, path, *, filename=None, device='cpu'):
        # recommended way from http://pytorch.org/docs/master/notes/serialization.html
        self.to(device)
        if not filename is None:
            path = os.path.join(path, filename)
        torch.save(self.state_dict(), path)


    def load(self, path, *, filename=None):
        if not filename is None:
            path = os.path.join(path, filename)
        with open(path, 'rb') as f:
            self.load_state_dict(torch.load(f))
            
            
class GANModel(object):
    """
    A wrapper class for Generator and Discriminator,
    'train_step' method is for single batch training.
    """

    fixed_noise = None
    fixed_labels = None
    criterion = None
    device = None
    netG = None
    netD = None
    optimizerG = None
    optimizerD = None
    nz = None
    nc = None
    ngf = None
    ndf = None
    num_classes = None
    real_cpu = None
    
    real = None
    fake = None
    
    def __init__(self, *, batch_size, nz, nc, ngf, ndf, num_classes, device, weights_init,
                    learning_rate, betas):

        super(GANModel, self).__init__()

        import torch
        
        self.device = device
        
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.num_classes = num_classes
        
        self.fixed_noise = torch.randn(batch_size, nz, 1, 1, device=self.device)
        self.fixed_labels = torch.LongTensor(
                torch.randint(0, self.num_classes, (batch_size,))).to(self.device)

        self.criterion = nn.BCELoss()
        
        self.real = torch.ones(batch_size).to(self.device)
        self.fake = torch.zeros(batch_size).to(self.device)

        self.netG = Generator(nz=nz, nc=nc, ngf=ngf, num_classes=self.num_classes).to(self.device)
        # print(netG)
        self.netD = Discriminator(nc=nc, ndf=ndf, num_classes=self.num_classes).to(self.device)
        # print(netD)
        
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        # setup optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=learning_rate, betas=betas)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=learning_rate, betas=betas)


    def train_step(self, images, labels, *, epoch, epochs):
        import torch

        ############################
        # (1) Data sampling
        ###########################
        self.real_cpu = images
        real_images = images.to(self.device)
        self.real_labels_cpu = labels
        real_labels = labels.to(self.device)
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake_labels = torch.LongTensor(
                torch.randint(0, self.num_classes, (batch_size,))).to(self.device)
        fake_images = self.netG(noise, fake_labels)
        
        
        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        self.netD.zero_grad()
        output = self.netD(real_images, real_labels).view(-1)
        errD_real = self.criterion(output, self.real)
        errD_real.backward()
        D_x = output.mean().item()


        # train with fake
        output = self.netD(fake_images.detach(), fake_labels).view(-1)
        errD_fake = self.criterion(output, self.fake)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()
    

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        # fake labels are real for generator cost
        output = self.netD(fake_images, fake_labels).view(-1)
        errG = self.criterion(output, self.real)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()


        return errG.item(), errD.item(), D_x, D_G_z1, D_G_z2
                
