import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train_vae(trainloader, testloader, net, epochs=10):
    opt = torch.optim.Adam(net.parameters())
    for epoch in range(epochs):
        for x, y in trainloader:
            x = x.to(device)
            opt.zero_grad
            x_hat = net(x)
            loss = ((x - x_hat)**2).sum() + net.encoder.kl
            loss.backward()
            opt.step()
    return net


