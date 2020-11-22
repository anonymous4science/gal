import argparse
import sys, os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.abspath(__file__), os.pardir, 'src'))
from lookahead import Lookahead
from adabound import AdaBound

# if you have installed Times New Roman on your machine, uncomment the following line
# rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
# rc('text', usetex=True)

cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def getWeights(parameters):
    weights = np.array([])
    for param in parameters():
        weights = np.concatenate((weights,param.data.cpu().view(-1).double().numpy()),axis=0)
    return weights

def getGrads(parameters):
    gradients = np.array([])
    for param in parameters():
        gradients = np.concatenate((gradients,param.grad.cpu().view(-1).double().numpy()),axis=0)
    return gradients

def getGrads_by_name(model, name=None):
    gradients = np.array([])
    for n_p, param in model.named_parameters():
        if n_p.startswith(name):
            gradients = np.concatenate((gradients,param.grad.cpu().view(-1).double().numpy()),axis=0)
    return gradients

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

fill_val=0.0
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(fill_val)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_net_outdim(indim, model):
    x_in = torch.randn(indim)
    outdim = torch.numel(model(x_in))
    return outdim

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ThirdPower(nn.Module):
    """docstring for ThirdPower"""
    def __init__(self, coef=0.1):
        super(ThirdPower, self).__init__()
        self.coef = coef

    def forward(self, x):
        x = self.coef*x**3
        return x

class RNet(nn.Module):
    def __init__(self, dim, out_planes=[]):
        super().__init__()
        self.dim = dim
        self.out_planes = out_planes
        net = []

        tmp_outs = out_planes
        in_channel = self.dim
        for i in range(0,len(tmp_outs)):
            net.append(nn.Linear(in_channel, tmp_outs[i]))
            in_channel = tmp_outs[i]
        net.append(nn.Linear(in_channel, self.dim))

        self.net = nn.Sequential(*net)

        self.net.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x

class SimpleModel(nn.Module):
    """
    A model to find a local minimum by gradient descent
    """
    def __init__(self, x, y,
        gal_ratio=0.0, gal_arch=None, gal_scale=1):
        super(SimpleModel, self).__init__()
        self.x = nn.Parameter(x)
        self.y = nn.Parameter(y)
        self.xy0 = np.array([x.item(),y.item()])

    def forward(self):
        _, _, z = cost_func([self.x, self.y])
        return z

    def process(self, optimizer):
        z = self.forward()
        x,y = self.x.item(), self.y.item()
        optimizer.zero_grad()
        z.backward()
        optimizer.step()

        return x, y, z.item()

class SimpleGALModel(nn.Module):
    """
    The proposed GAL method embedded in a basic model
    """
    def __init__(self, x, y, 
        gal_ratio=0.0, gal_arch=None, gal_scale=1):
        super(SimpleGALModel, self).__init__()
        self.xy = nn.Parameter(torch.tensor([x,y]))
        self.gal_scale = gal_scale
        self.gal_arch = gal_arch
        self.gal_ratio=gal_ratio
        self.xy0 = np.array([x.item(),y.item()])

        self.rnet = RNet(2, gal_arch)
        print('# of parameters of GradPredictor: {}'.format(count_parameters(self.rnet)))

    def forward(self,x):
        _, _, z = cost_func(x)
        
        return z

    def process(self, optimizer):

        tmp_xy = Variable(torch.FloatTensor(self.xy.detach().numpy()),requires_grad=True)
        z = self.forward(tmp_xy)
        x,y = self.xy[0].item(), self.xy[1].item()
        optimizer.zero_grad()
        # compute first-order derivatives by backpropagation
        z.backward()
        grad = tmp_xy.grad

        # compute learning rate for tentative loss
        cur_lr = optimizer.param_groups[0]['lr']*self.gal_scale

        # predict the gradient adjustment
        adjustment = self.rnet(self.xy)
        # nomalize the adjustment
        adjustment = adjustment / \
                        adjustment.norm() * \
                        grad.norm()
        # adjust the gradient
        delta_grad = cur_lr*(grad+self.gal_ratio*adjustment)
        # compute the tentative loss
        _,_,z_tentative = cost_func(self.xy - delta_grad)
        # train h() by minimizing the remainder
        first_term = (delta_grad*grad).mean()
        loss_remainder = torch.abs(z_tentative + first_term - z.item())
        optimizer.zero_grad()
        loss_remainder.backward()

        if z_tentative <= z:
            grad_for_update = grad + self.gal_ratio*adjustment.data
        else:
            grad_for_update = grad

        self.xy.grad.data.copy_(grad_for_update)

        optimizer.step()

        return x, y, z.item()


def cost_func(xy=None):
    '''Cost function.

    Args:
        xy: a list or tensor containing x and y.

    Returns:
        Tuple (x, y, z) where x and y are input tensors and z is output tensor.
    '''

    # two local minima near (0, 0)
    x = xy[0]
    y = xy[1]
    z = -1 * torch.sin(x * x) * torch.cos(3 * y * y) * torch.exp(-(x * y) * (x * y)) - torch.exp(-(x + y) * (x + y))

    return x, y, z

# pyplot settings
plt.ion()
fig = plt.figure(figsize=(3, 2), dpi=200)
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0, bottom=0, right=1, top=0.938, wspace=0, hspace=0)
params = {'legend.fontsize': 5,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.axis('off')

# starting location for variables
x_i = 0.75
y_i = 1.0
steps = 300
xy_range = [-1.5, 1.5]

# visualize cost function as a contour plot
x_val = y_val = np.arange(xy_range[0], xy_range[1], 0.005, dtype=np.float32)
x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
y_val_mesh_flat = y_val_mesh.reshape([-1, 1])

_, _, z_val_mesh_flat = cost_func([torch.from_numpy(x_val_mesh_flat), torch.from_numpy(y_val_mesh_flat)])
z_val_mesh_flat = z_val_mesh_flat.data.numpy()

z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)
levels = np.arange(np.min(z_val_mesh_flat), np.max(z_val_mesh_flat), 0.05)
ax.contour(x_val_mesh, y_val_mesh, z_val_mesh, levels, alpha=.7, linewidths=0.4)
plt.draw()

# 3d plot
xlm = ax.get_xlim3d()
ylm = ax.get_ylim3d()
zlm = ax.get_zlim3d()
ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
azm = ax.azim
ele = ax.elev + 40
ax.view_init(elev=ele, azim=azm)

parser = argparse.ArgumentParser(description='Convergence Visualization for the DCL Work')
parser.add_argument('--opt', type=str, default='gd', 
                    help='Optimizer type: gd, lagd, rmsprop, adam, adabound')
args = parser.parse_args()

manualSeed = random.randint(1, 10000)
if args.opt == 'gd':
    manualSeed = 2679
elif args.opt == 'lagd':
    manualSeed = 4870
elif args.opt == 'rmsprop':
    manualSeed = 9910
elif args.opt == 'adam':
    manualSeed = 9049
elif args.opt == 'adabound':
    manualSeed = 5026
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
print('===Seed: {}'.format(manualSeed))

img_suffix = 'png' # image extension: e.g., 'svg', 'png'
z_file = 'z_results.npz'
output_path = '{}_{}'.format('figures', args.opt)
if not os.path.exists(output_path):
    os.makedirs(output_path)

arch = [64,24]
if args.opt == 'gd':
    steps = 100
    lr = 0.1
    gal_scale = 1
    gal_ratio = 0.2
    ops_param = np.array([[optim.SGD, lr, 'b', 'GD',     SimpleModel,    0.5, 0.5, gal_ratio, None, gal_scale, '-'],
                        [optim.SGD,   lr, 'r', 'GD GAL', SimpleGALModel, 0.5, 0.5, gal_ratio, arch, gal_scale, '-']])
elif args.opt == 'lagd':
    steps = 600
    lr = 0.01
    gal_scale = 1
    gal_ratio = 0.1
    ops_param = np.array([[optim.SGD, lr, 'b', 'LA(GD)',     SimpleModel,    0.5, 0.5, gal_ratio, None, gal_scale, '-'],
                        [optim.SGD,   lr, 'r', 'LA(GD) GAL', SimpleGALModel, 0.5, 0.5, gal_ratio, arch, gal_scale, '-']])
elif args.opt == 'rmsprop':
    steps = 150
    lr = 0.02
    gal_scale = 1
    gal_ratio = 0.3
    ops_param = np.array([[optim.RMSprop, lr, 'b', 'RMSP',     SimpleModel,    0.0, 0.0, gal_ratio, None, gal_scale, '-'],
                        [optim.RMSprop,   lr, 'r', 'RMSP GAL', SimpleGALModel, 0.0, 0.0, gal_ratio, arch, gal_scale, '-']])
elif args.opt == 'adam':
    steps = 250
    lr = 0.02
    gal_scale = 0.1
    gal_ratio = 0.4
    ops_param = np.array([[optim.Adam, lr, 'b', 'Adam',     SimpleModel,    0.0, 0.0, gal_ratio, None, gal_scale, '-'],
                        [optim.Adam,   lr, 'r', 'Adam GAL', SimpleGALModel, 0.0, 0.0, gal_ratio, arch, gal_scale, '-']])
elif args.opt == 'adabound':
    steps = 300
    lr = 0.4
    gal_scale = 1
    gal_ratio = 0.5
    ops_param = np.array([[AdaBound, lr, 'b', 'ADB',     SimpleModel,    0.2, 0.0, gal_ratio, None, gal_scale, '-'],
                        [AdaBound,   lr, 'r', 'ADB GAL', SimpleGALModel, 0.2, 0.0, gal_ratio, arch, gal_scale, '-']])

method_names = []
for i in range(ops_param.shape[0]):
    method_names.append(ops_param[i,3])
method_names = np.array(method_names)

if not os.path.exists(output_path):
    os.makedirs(output_path)

# use last location to draw a line to the current location
last_x, last_y, last_z = [], [], []
plot_cache = [None for _ in range(ops_param.shape[0])]
plot_line_cache = [None for _ in range(ops_param.shape[0])]

min_x, min_y, min_z = sys.float_info.max,sys.float_info.max,sys.float_info.max
best_method = ''
models = []
optimizers = []
save_results = np.full((ops_param.shape[0],steps), sys.float_info.max)
for i in range(ops_param.shape[0]):
    x = torch.tensor(x_i)
    y = torch.tensor(y_i)
    sm = ops_param[i,4](x, y, gal_ratio=ops_param[i,7], gal_arch=ops_param[i,8], gal_scale=ops_param[i,9])
    
    models.append(sm)
    if args.opt == 'gd':
        optimizers.append(ops_param[i,0](filter(lambda p: p.requires_grad, sm.parameters()), lr=ops_param[i,1]))
    elif args.opt == 'lagd':
        optimizer = ops_param[i,0](filter(lambda p: p.requires_grad, sm.parameters()), lr=ops_param[i,1])
        lookahead = Lookahead(optimizer, k=5, alpha=0.5)
        optimizers.append(lookahead)
    elif args.opt == 'rmsprop':
        optimizers.append(ops_param[i,0](filter(lambda p: p.requires_grad, sm.parameters()), 
            lr=ops_param[i,1], momentum=ops_param[i,5], weight_decay=ops_param[i,6], eps=1e-01))
    elif args.opt == 'adam':
        optimizers.append(ops_param[i,0](filter(lambda p: p.requires_grad, sm.parameters()), 
            lr=ops_param[i,1], betas=(0.8,0.999), eps=1e-1, weight_decay=ops_param[i,6]))
    elif args.opt == 'adabound':
        optimizers.append(ops_param[i,0](filter(lambda p: p.requires_grad, sm.parameters()), 
                            lr=ops_param[i,1],
                            final_lr=0.02, 
                            gamma=5e-1))

for iter in range(steps):
    for i, optimizer in enumerate(optimizers):
        x_val, y_val, z_val = models[i].process(optimizer)
        save_results[i, iter] = z_val
        if min_z > z_val:
            min_x, min_y, min_z = x_val, y_val, z_val
            best_method = ops_param[i,3]

        if plot_cache[i]:
            plot_cache[i].remove()
        plot_cache[i] = ax.scatter(x_val, y_val, z_val, s=3, depthshade=True, label=ops_param[i, 3], color=ops_param[i, 2])
        # draw a line from the previous value
        if iter == 0:
            last_z.append(z_val)
            last_x.append(x_i)
            last_y.append(y_i)
        plot_line_cache[i], = ax.plot([last_x[i], x_val], [last_y[i], y_val], [last_z[i], z_val], linewidth=0.5, linestyle=ops_param[i, -1], color=ops_param[i, 2], label=ops_param[i, 3])
        last_x[i] = x_val
        last_y[i] = y_val
        last_z[i] = z_val

    if iter % 50 == 0:
        np.savez(os.path.join(output_path, z_file), z=save_results, names=method_names)

    if iter == 0:
        legend = ops_param[:, 3]
        plt.legend(plot_line_cache, legend, 
                bbox_to_anchor=(0.5,1.085), frameon=False,
                loc=9, ncol=2)

    if img_suffix == 'png' or img_suffix == 'jpg':
        plt.savefig('{}.{}'.format(os.path.join(output_path,str(iter)), img_suffix), dpi=400)
    else:
        plt.savefig(os.path.join(output_path,str(iter) + '.svg'))
    print('iteration: {}, ({},{},{}), {}'.format(iter, min_x, min_y, min_z, best_method))

    plt.pause(0.0001)
np.savez(os.path.join(output_path, z_file), z=save_results, names=method_names)
print("done")
