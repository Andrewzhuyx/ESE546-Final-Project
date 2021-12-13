import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#TODO: Create ib_encoder

class vae_encoder(nn.Module):
    def __init__(self, n_layers, in_dim=784, z_dim=32, kernel_size=3, stride=2):
        super(vae_encoder, self).__init__()
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.in_ch = [2**k for k in range(n_layers)]
        self.layers = nn.ModuleList([nn.Sequential(nn.Conv2d(ch, ch*2, 
                                                   kernel_size=kernel_size, 
                                                   stride=stride, padding=0),
                                                   nn.ReLU())
                                     for ch in self.in_ch])
        self.mu = nn.Linear(int(np.floor(np.sqrt(in_dim // 4**n_layers))) * 2**n_layers, z_dim) 
        self.logvar = nn.Linear(int(np.floor(np.sqrt(in_dim // 4**n_layers))) * 2**n_layers, z_dim)
        self.dist = torch.distributions.Normal(0, 1)
        # CUDA hack for above?
        # https://avandekleut.github.io/vae/
        print("Encoder Layers")
        print(self.layers)
        print(self.mu)
        print(self.logvar,'\n')
                
    def forward(self, x):
        x = self.layers(x)
        mu = F.ReLU(self.mu(x))
        logvar = F.ReLU(self.logvar(x))
        z = mu + torch.exp(logvar) * self.dist.sample(mu.shape)
        self.kl = torch.exp(logvar)**2 + mu**2 - torch.log(torch.exp(logvar) - 1/2).sum()
        return z
         

class decoder(nn.Module):
    def __init__(self, n_layers, out_dim=784, z_dim=32, kernel_size=3, stride=2):
        super(decoder, self).__init__()
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.z_dim = z_dim
        self.in_ch = int(np.floor(np.sqrt(out_dim // 4**n_layers))) * 2**n_layers
        self.fc = nn.Linear(z_dim, self.in_ch)         
        self.in_ch = [self.in_ch // 2**k for k in range(n_layers)]
        self.layers = nn.ModuleList([nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)),
                                                   nn.ConvTranspose2d(ch, ch // 2,
                                                                      kernel_size=kernel_size,
                                                                      stride=stride),
                                                   nn.ReLU()) 
                                     if n_layers >= 3 and layer == n_layers - 3 
                                     else nn.Sequential(nn.ConvTranspose2d(ch, ch // 2, kernel_size=kernel_size,
                                                                           stride=stride), nn.ReLU()) for layer, ch in enumerate(self.in_ch)])
        print("Decoder Layers:")
        print(self.fc)
        print(self.layers, '\n')

    def forward(self, x):
        x = self.fc(x)
        side_dim = np.floor(np.sqrt(self.out_dim // 4**n_layers))
        x = x.reshape(x[0], side_dim, side_dim, -1)
        x = self.layers(x)
        return x


class variational_autoencoder(nn.Module):
    def __init__(self, n_layers, in_dim=784, z_dim=32, kernel_size=3, stride=2):
        super(variational_autoencoder, self).__init__()
        self.encoder = vae_encoder(n_layers, in_dim, z_dim, kernel_size, stride)
        self.decoder = decoder(n_layers, in_dim, z_dim, kernel_size, stride)

    def encoder(self, x):
        return encoder(x)

    def decode(self, z):
        return decoder(z)

    def forward(self, x):
        z = encoder(x)
        x_hat = decoder(z)
        return x_hat



#test_enc = encoder(4)
#test_dec = decoder(4) 
test_vae = variational_autoencoder(4)
