import numpy as np
import torch
import torch.nn as nn
import random
import os

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

nz = 100
ngf = 64
nc=3
device = torch.device("cpu")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Model:
    def __init__(self, filename=None, model=None):
        """
        """
        # make torch seed depend on system seed
        manSeed = random.randint(0, 999999999)
        self.netG = Generator().to(device)
        self.netG.apply(weights_init)
        if filename is not None:
            self.netG.load_state_dict(torch.load(filename, map_location='cpu'))
        print(self.netG)

    def encode_images(self, images):
        """
        Encode images x => z

        images is an n x 3 x s x s numpy array where:
          n = number of images
          3 = R G B channels
          s = size of image (eg: 64, 128, etc)
          pixels values for each channel are encoded [0,1]

        returns an n x z numpy array where:
          n = len(images)
          z = dimension of latent space
        """
        # todo
        pass

    def get_zdim(self):
        """
        Returns the integer dimension of the latent z space
        """
        return 100

    def sample_at(self, z):
        """
        Decode images z => x

        z is an n x z numpy array where:
          n = len(images)
          z = dimension of latent space

        return images as an n x 3 x s x s numpy array where:
          n = number of images
          3 = R G B channels
          s = size of image (eg: 64, 128, etc)
          pixels values for each channel are encoded [0,1]
        """
        z_len, nz = z.shape
        z = z.reshape(z_len, nz, 1, 1)
        # fixed_noise = torch.randn(z_len, nz, 1, 1, device=device)
        # fix1 = fixed_noise.numpy()
        # print(fix1.shape, " VERSUS ", z.shape)
        fake = self.netG(torch.from_numpy(z).float())

        # fake = self.netG(torch.from_numpy(z))
        f = (fake.detach().numpy() + 1.0) / 2.0
        print(f.shape)
        print(np.amin(f))
        print(np.amax(f))
        return f
        # return fake.detach().permute(1, 2, 0).to('cpu').numpy()
        # print(decoded.shape)
        # channel_first = np.rollaxis(decoded, 3, 1)  
        # print(channel_first.shape)
        # return channel_first
