'''Train CIFAR10 with PyTorch.
mostly from  https://github.com/zhirongw/lemniscate.pytorch/blob/master/cifar.py, AET
'''
from __future__ import print_function

import sys
import os
import argparse
import time

from asano import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as tfs
from tensorboardX import SummaryWriter

from asano.util import AverageMeter, setup_runtime, py_softmax
from asano.cifar_utils import kNN, CIFAR10Instance, CIFAR100Instance

from src.ml.sinkhorn import SinkhornValue, sinkhorn, pot_sinkhorn


def feature_return_switch(model, bool=True):
    """
    switch between network output or conv5features
        if True: changes switch s.t. forward pass returns post-conv5 features
        if False: changes switch s.t. forward will give full network output
    """
    if bool:
        model.headcount = 1
    else:
        model.headcount = args.hc
    model.return_features = bool


parser = argparse.ArgumentParser(description='PyTorch Implementation of Self-Label for CIFAR10/100')

parser.add_argument('--device', default="1", type=str, help='cuda device')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--restart', action='store_true', help='restart opt')

# model
parser.add_argument('--arch', default='alexnet', type=str, help='architecture')
parser.add_argument('--ncl', default=256, type=int, help='number of clusters')
parser.add_argument('--hc', default=10, type=int, help='number of heads')

# SK-optimization
#parser.add_argument('--lamb', default=10.0, type=float,                   help='SK lambda parameter')
#parser.add_argument('--nopts', default=400, type=int, help='number of SK opts')

# optimization
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='sgd momentum')
parser.add_argument('--epochs', default=800, type=int,
                    help='number of epochs to train')
parser.add_argument('--batch-size', default=1024, type=int,
                    metavar='BS', help='batch size')

# logging saving etc.
parser.add_argument(
    '--datadir', default='/home/mlelarge/data', type=str)
parser.add_argument('--exp', default='/home/mlelarge/GitHub/SiMaC/expe', type=str, help='experimentdir')
parser.add_argument('--type', default='10', type=int, help='cifar10 or 100')

args = parser.parse_args()
# setup_runtime(2, [args.device])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
knn_dim = 4096
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
# Data
print('==> Preparing data..')
transform_train = tfs.Compose([
    tfs.Resize(256),
    tfs.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    tfs.ColorJitter(0.4, 0.4, 0.4, 0.4),
    tfs.RandomGrayscale(p=0.2),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = tfs.Compose([
    tfs.Resize(256),
    tfs.CenterCrop(224),
    tfs.ToTensor(),
    tfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.type == 10:
    trainset = CIFAR10Instance(root=args.datadir, train=True, download=True,
                               transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = CIFAR10Instance(root=args.datadir, train=False, download=True,
                              transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
else:

    trainset = CIFAR100Instance(root=args.datadir, train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = CIFAR100Instance(root=args.datadir, train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)


print('==> Building model..')
numc = [args.ncl] * args.hc
model = models.__dict__[args.arch](num_classes=numc)
knn_dim = 4096

N = len(trainloader.dataset)
#optimize_times = ((args.epochs + 1.0001)*N*(np.linspace(0, 1, args.nopts))[::-1]).tolist()
#optimize_times = [(args.epochs +10)*N] + optimize_times
#print('We will optimize L at epochs:', [np.round(1.0*t/N, 2) for t in optimize_times], flush=True)

# init selflabels randomly
if args.hc == 1:
    selflabels = np.zeros(N, dtype=np.int32)
    for qq in range(N):
        selflabels[qq] = qq % args.ncl
    selflabels = np.random.permutation(selflabels)
    selflabels = torch.LongTensor(selflabels).cuda()
else:
    selflabels = np.zeros((args.hc, N), dtype=np.int32)
    for nh in range(args.hc):
        for _i in range(N):
            selflabels[nh, _i] = _i % numc[nh]
        selflabels[nh] = np.random.permutation(selflabels[nh])
    selflabels = torch.LongTensor(selflabels).cuda()


optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=5e-4)
# Model
if args.test_only or len(args.resume) > 0:
    # Load checkpoint.[
    print('==> Resuming from checkpoint..')
    assert(os.path.isdir('%s/' % (args.exp)))
    checkpoint = torch.load(args.resume)
    print('loaded checkpoint at: ', checkpoint['epoch'])
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    if 'opt' in list(checkpoint.keys()):
        optimizer.load_state_dict(checkpoint['opt'])
    selflabels = checkpoint['L']
    selflabels = selflabels.to(device)
    include = [(qq / N >= start_epoch) for qq in optimize_times]
    optimize_times = (np.array(optimize_times)[include]).tolist()
    # print('We will optimize L at epochs:', [np.round(1.0 * t / N, 2) for t in optimize_times], flush=True)
    model.to(device)
    # for state in optimizer.state.values():
    #     for k, v in state.items():
    #         if isinstance(v, torch.Tensor):
    #             state[k] = v.cuda()


model.to(device)
#criterion = nn.CrossEntropyLoss()

if args.test_only:
    feature_return_switch(model, True)
    usepca = True
    acc = kNN(model, trainloader, testloader, K=[200, 50, 10, 5, 1], sigma=[
              0.1, 0.5], dim=knn_dim, use_pca=usepca)
    sys.exit(0)

name = "%s" % args.exp.replace('/', '_')
writer = SummaryWriter(f'./runs/cifar{args.type}/{name}')
writer.add_text('args', " \n".join(
    ['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))



# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    print(name)
    #adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        #niter = epoch * len(trainloader) + batch_idx
        # if niter * trainloader.batch_size >= optimize_times[-1]:
        #     with torch.no_grad():
        #         _ = optimize_times.pop()
        #         if args.hc >1:
        #             feature_return_switch(model, True)
        #         selflabels = opt_sk(model, selflabels, epoch)
        #         if args.hc >1:
        #             feature_return_switch(model, False)
        data_time.update(time.time() - end)
        inputs, targets, indexes = inputs.to(
            device), targets.to(device), indexes.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        #M = outputs
        #M = (outputs - np.log(inputs.shape[0])).to(device)

        # # marginals
        # minibatch size
        a = (torch.ones(inputs.shape[0]) / inputs.shape[0]).to(device)
        # K clusters
        b = (torch.ones(args.ncl) / args.ncl).to(device)

        SV = SinkhornValue(
            a,
            b,
            epsilon=.04,
            solver=pot_sinkhorn
            #numIterMax=400
        )

        if args.hc == 1:
            # loss = criterion(outputs, selflabels[indexes])
            loss = SV(-outputs)
        else:
            #loss = torch.mean(torch.stack([criterion(outputs[h], selflabels[h, indexes]) for h in range(args.hc)]))
            loss = torch.mean(torch.stack([SV(-outputs[h]) for h in range(args.hc)]))

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 10 == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
                epoch, batch_idx, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))
            writer.add_scalar("loss", loss.item(), batch_idx*512 +epoch*len(trainloader.dataset))
    pass


for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    feature_return_switch(model, True)
    acc = kNN(model, trainloader, testloader, K=10, sigma=0.1, dim=knn_dim)
    feature_return_switch(model, False)
    writer.add_scalar("accuracy kNN", acc, epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'opt': optimizer.state_dict(),
            'L': selflabels,
        }
        if not os.path.isdir(args.exp):
            os.mkdir(args.exp)
        torch.save(state, '%s/best_ckpt.t7' % (args.exp))
        best_acc = acc
    if epoch % 100 == 0:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'opt': optimizer.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'L': selflabels,
        }
        if not os.path.isdir(args.exp):
            os.mkdir(args.exp)
        torch.save(state, '%s/ep%s.t7' % (args.exp, epoch))
    if epoch % 50 == 0:
        feature_return_switch(model, True)
        acc = kNN(model, trainloader, testloader, K=[50, 10],
                  sigma=[0.1, 0.5], dim=knn_dim, use_pca=True)
        i = 0
        for num_nn in [50, 10]:
            for sig in [0.1, 0.5]:
                writer.add_scalar('knn%s-%s' % (num_nn, sig), acc[i], epoch)
                i += 1
        feature_return_switch(model, False)
    print('best accuracy: {:.2f}'.format(best_acc * 100))

checkpoint = torch.load('%s' % args.exp+'/best_ckpt.t7')
model.load_state_dict(checkpoint['net'])
feature_return_switch(model, True)
acc = kNN(model, trainloader, testloader, K=10,
          sigma=0.1, dim=knn_dim, use_pca=True)
