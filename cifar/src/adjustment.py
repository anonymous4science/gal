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
    def __init__(self, dim, gal_arch=[64]):
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

class Optimization_wrapper(nn.Module):
    def __init__(self, method='standard', 
            gal_ratio=0.1):
        super(Optimization_wrapper, self).__init__()
        self.method = method
        self.gal_ratio = gal_ratio

    def process(self, criterion, x, target, regressor, lr=0.1):
        method = self.method
        loss_val = -1

        if method == 'standard':
            loss = criterion(x, target)
            loss.backward()            
            loss_val = loss.item()
        elif method == 'gal':
            cur_lr = lr
            x_shadow = Variable(torch.FloatTensor(x.data.cpu().numpy()).cuda(), requires_grad=True)
            
            loss = criterion(x_shadow, target)
            loss_val = loss.item()
            # compute first-order derivatives by backpropagation
            loss.backward()
            x_grad = getGrads(x_shadow)
            x_grad = Variable(torch.FloatTensor(x_grad).cuda(), requires_grad=True)

            # predict the gradient adjustment
            pred_adj = regressor(x_shadow)
            # nomalize the adjustment
            pred_adj = self.gal_ratio*pred_adj / \
                        pred_adj.norm(dim=1, keepdim=True) * \
                        x_grad.norm(dim=1, keepdim=True)
            # adjust the gradient
            adjusted_grad = x_grad+pred_adj 
            delta_grad = adjusted_grad*cur_lr
            x_updated = x_shadow - delta_grad
            # compute the tentative loss
            loss_tentative = criterion(x_updated, target)
            # train h() by minimizing the remainder
            first_term = (delta_grad*x_grad).sum(1)
            loss_remainder = torch.abs(loss_tentative + first_term.mean() - loss_val)
            loss_remainder.backward()
            
            if loss_tentative.item() <= loss.item():
                grad_for_update = x_grad.data + pred_adj.data
            else:
                grad_for_update = x_grad.data
            x.backward(grad_for_update)

        return loss_val