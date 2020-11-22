import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def getGrads(x):
    gradients = x.grad.cpu().double().numpy()
    return gradients

def getGrads_by_name(model, name=None):
    gradients = np.array([])
    for n_p, param in model.named_parameters():
        if n_p.startswith(name):
            gradients = np.concatenate((gradients,param.grad.cpu().view(-1).double().numpy()),axis=0)
    return gradients

def zeroTensorGrads(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_net_outdim(indim, model):
    x_in = torch.randn(1,indim)
    outdim = torch.numel(model(x_in))
    return outdim

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class RNet(nn.Module):
    def __init__(self, dim, gal_arch=[512]):
        super().__init__()
        self.dim = dim
        self.gal_arch = gal_arch

        self.net = None
        if gal_arch is not None:
            net = []
            tmp_outs = gal_arch

            needBias = True
            stride_val = 2
            kernal_size = 3
            
            in_channel = self.dim
            for i in range(0,len(gal_arch)):
                net.append(nn.Linear(in_channel, gal_arch[i]))
                in_channel = gal_arch[i]
            net.append(nn.Linear(in_channel, self.dim))

            self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x
