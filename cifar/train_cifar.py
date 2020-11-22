from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import subprocess

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig

import sys
srcFolder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'src')
sys.path.append(srcFolder)
from adjustment import *
from lookahead import Lookahead
from adabound import AdaBound

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (get_model_params, BlockDecoder)

cudnn.benchmark = True

class PrintLogger(object):
    def __init__(self, log_path, mode='a'):
        self.terminal = sys.stdout
        self.log = open(log_path, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        self.log.flush()
        self.terminal.flush()
        pass

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(m.weight)
        # nn.init.kaiming_normal_(m.weight)
        # m.weight.data.normal_(0.0, 0.2)
        # nn.init.xavier_uniform_(m.weight.data)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--arch', '-a', default='efficientnet-b1',
                    type=str,
                    help='model architecture: resnet101 | efficientnet-b1')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer: sgd|lookahead|adabound')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--datapath', type=str, default='', help='the path to the datasets')
parser.add_argument('--mtype', type=str, default='standard', help='method type: standard|gal')
parser.add_argument('--gal_scale', type=float, default=1, help='lr scale used in GAL')
parser.add_argument('--gal_ratio', type=float, default=0.001, help='magnitude ratio used in GAL')
parser.add_argument('--gal_arch', nargs='+', type=int, help='arch for adjustment learning')
parser.add_argument('--optim_adabound_final_lr', type=float, default=0.1, help='final lr used in optimizer AdaBound')
parser.add_argument('--optim_adabound_beta1', type=float, default=0.9, help='beta1 used in optimizer AdaBound')
parser.add_argument('--optim_adabound_beta2', type=float, default=0.999, help='beta2 used in optimizer AdaBound')
parser.add_argument('--optim_adabound_gamma', type=float, default=1e-3, help='gamma used in optimizer AdaBound')
parser.add_argument('--optim_lookahead_k', type=int, default=5, help='K used in optimizer Lookahead')
parser.add_argument('--optim_lookahead_alpha', type=float, default=.8, help='alpha used in optimizer Lookahead')
parser.add_argument('--optim_lookahead_beta1', type=float, default=0.9, help='beta1 used in optimizer Lookahead')
parser.add_argument('--optim_lookahead_beta2', type=float, default=0.999, help='beta2 used in optimizer Lookahead')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
# print('==> Seed: {}'.format(args.manualSeed))

best_acc = 0  # best test accuracy
best_epoch = 0

def print_to(input_str, file, append=True):
    if append:
        flag = 'a'
    else:
        flag = 'w'
    with open(file, flag) as f:
            print(input_str, file=f, flush=True)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    training_log = os.path.join(args.checkpoint, 'log.txt')
    sys.stdout = PrintLogger(training_log, 'w')
    print('==> Seed: {}'.format(args.manualSeed))
    # print_to(args.manualSeed, training_log, append=False)

    # Data
    print('==> Preparing dataset {}'.format(args.dataset))
    # print_to('==> Preparing dataset {}'.format(args.dataset), training_log)
    if args.arch.startswith('efficientnet'):
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    trainset = dataloader(root=args.datapath, train=True, download=False, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root=args.datapath, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    # print_to("==> creating model '{}'".format(args.arch), training_log)
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )
    elif args.arch.startswith('efficientnet'):
        model = EfficientNet.from_pretrained(args.arch, num_classes=num_classes)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    modeltype = type(model).__name__

    model = torch.nn.DataParallel(model).cuda()
    regressor = None
    if args.mtype == 'gal':
        # instantiate a regression net for gradient adjustment prediction
        regressor = RNet(num_classes, args.gal_arch)
        regressor = torch.nn.DataParallel(regressor).cuda()

    if regressor is not None:
        print(regressor)
        output_str = '====>Total params: {:.2f}M + {}'.format(
            sum(p.numel() for p in model.parameters())/1000000.0,
            sum(p.numel() for p in regressor.parameters()))
        print(output_str)
    else:
        output_str = '====>Total params: {:.2f}M'.format(
            sum(p.numel() for p in model.parameters())/1000000.0)
        print(output_str)

    criterion = nn.CrossEntropyLoss()

    if regressor is None:
        learnable_param = model.parameters()
    else:
        learnable_param = [
                {'params': model.parameters()},
                {'params': regressor.parameters()}
            ]
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(learnable_param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'lookahead':
        optimizer = optim.SGD(learnable_param, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer, k=args.optim_lookahead_k, alpha=args.optim_lookahead_alpha)
    elif args.optimizer == 'adabound':
        optimizer = AdaBound(learnable_param, lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.optim_adabound_beta1,args.optim_adabound_beta2),
            final_lr=args.optim_adabound_final_lr, gamma=args.optim_adabound_gamma)

    optim_wrapper = Optimization_wrapper(method=args.mtype, gal_ratio=args.gal_ratio)
    optim_wrapper = optim_wrapper.cuda()

    print(args)
    
    # Resume
    title = 'cifar-10-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.mtype == 'gal':
            regressor.load_state_dict(checkpoint['gal_state_dict'])
        logger = Logger(os.path.join(args.checkpoint, 'stat.csv'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'stat.csv'), title=title)
        logger.set_names(['Train Loss', 'Valid Loss', 'Train Proc Time', 'Valid Proc Time', 'Train Err.', 'Valid Err.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    pull_gpu_usage = True
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_top1_acc, train_top5_acc, train_misc = train(trainloader, model, criterion, optimizer, epoch, use_cuda, optim_wrapper, args.gal_scale, regressor)
        test_loss, test_top1_acc, test_top5_acc, test_misc = test(testloader, model, criterion, epoch, use_cuda)
        train_proc_time = train_misc[0]
        test_proc_time = test_misc[0]

        print('train_loss: {:.4f}, train_top1_err: {:.2f}, train_top5_err: {:.2f}, test_loss: {:.4f}, test_top1_err: {:.2f}, test_top5_err: {:.2f}'.format(
            train_loss, 100-train_top1_acc, 100-train_top5_acc,
            test_loss, 100-test_top1_acc, 100-test_top5_acc))

        # append logger file
        logger.append([train_loss, test_loss, train_proc_time, test_proc_time, 100-train_top1_acc, 100-test_top1_acc])

        # save model
        is_best = test_top1_acc > best_acc
        best_acc = max(test_top1_acc, best_acc)
        if is_best:
            best_epoch = epoch + 1
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'gal_state_dict': regressor.state_dict() if args.mtype == 'gal' else None,
                'acc': test_top1_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()

    lowest_err = 100-best_acc
    print('Best err: {:.04f} at {}-th epoch'.format(100-best_acc, best_epoch))

    renamed_folder = '{}_{:.02f}'.format(args.checkpoint.replace(' ', '_'), lowest_err)
    os.rename(args.checkpoint, renamed_folder)

def train(trainloader, model, criterion, optimizer, epoch, 
        use_cuda, optim_wrapper=None, gal_scale=1.0, regressor=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    proc_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    misc = []
    losses_std, losses_gg_std = [], []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        ptime_start = time.time()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        optimizer.zero_grad()
        loss_val = optim_wrapper.process(criterion, outputs, targets, 
                        regressor, lr=optimizer.param_groups[0]['lr']*gal_scale)
        optimizer.step()

        proc_time.update(time.time() - ptime_start, inputs.size(0))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss_val, inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    misc.append(proc_time.avg)
    return (losses.avg, top1.avg, top5.avg, misc)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    proc_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    misc = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        ptime_start = time.time()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss = loss.mean()
            loss_val = loss.item()

        proc_time.update(time.time() - ptime_start)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    misc.append(proc_time.avg)
    return (losses.avg, top1.avg, top5.avg, misc)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*args.gamma

if __name__ == '__main__':
    main()
