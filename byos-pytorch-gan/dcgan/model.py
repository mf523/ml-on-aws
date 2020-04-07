import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import os

import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, *, nz, nc, ngf):
        super(Generator, self).__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(nz, ngf*8,
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


    def forward(self, x):
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
    def __init__(self, *, nc, ndf):
        super(Discriminator, self).__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(nc, ndf,
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(ndf, ndf*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf*2, ndf*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf*4, ndf*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False)
        

    def forward(self, x):
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
            
            
class DCGAN(object):
    """
    A wrapper class for Generator and Discriminator,
    'train_step' method is for single batch training.
    """

    fixed_noise = None
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
    real_cpu = None
    
    def __init__(self, *, batch_size, nz, nc, ngf, ndf, device, weights_init,
                    learning_rate, betas, real_label, fake_label):

        super(DCGAN, self).__init__()

        import torch
        
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        
        self.real_label = real_label
        self.fake_label = fake_label
        
        self.fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
        self.criterion = nn.BCELoss()
        self.device = device
        
        self.netG = Generator(nz=nz, nc=nc, ngf=ngf).to(device)
        # print(netG)
        self.netD = Discriminator(nc=nc, ndf=ndf).to(device)
        # print(netD)
        
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)
        
        # setup optimizer
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=learning_rate, betas=betas)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=learning_rate, betas=betas)


    def train_step(self, data, *, epoch, epochs, batch, batches):
        import torch

        ############################
        # (1) Data sampling
        ###########################
        self.real_cpu = data[0]
        real = data[0].to(self.device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        fake = self.netG(noise)
        
        
        ############################
        # (2) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        self.netD.zero_grad()
        label = torch.full((batch_size,), self.real_label, device=self.device)
        output = self.netD(real).view(-1)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()


        # train with fake
        label.fill_(self.fake_label)
        output = self.netD(fake.detach()).view(-1)
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()
    

        ############################
        # (3) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        output = self.netD(fake).view(-1)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()


        return errG.item(), errD.item(), D_x, D_G_z1, D_G_z2
                
