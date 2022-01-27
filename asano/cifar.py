"""Train CIFAR10 with PyTorch.
mostly from  https://github.com/zhirongw/lemniscate.pytorch/blob/master/cifar.py, AET
"""
from __future__ import print_function

import os
import argparse
import time
import logging

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as tfs
from tensorboardX import SummaryWriter

from asano import models
from asano.util import AverageMeter
from asano.cifar_utils import kNN, CIFAR10Instance, CIFAR100Instance
from src.ml.sinkhorn import SinkhornValue, pot_sinkhorn


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s]: %(message)s"
)

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


parser = argparse.ArgumentParser(
    description="PyTorch Implementation of Self-Label for CIFAR10/100")

parser.add_argument("--device", default="1", type=str,
                    help="cuda device")
parser.add_argument("--resume", "-r", default="", type=str,
                    help="resume from checkpoint")
parser.add_argument("--test-only", action="store_true",
                    help="test only")
parser.add_argument("--restart", action="store_true",
                    help="restart opt")

# model
parser.add_argument("--arch", default="alexnet", type=str,
                    help="architecture")
parser.add_argument("--ncl", default=256, type=int,
                    help="number of clusters")
parser.add_argument("--hc", default=10, type=int,
                    help="number of heads")

# SK-optimization
parser.add_argument("--lamb", default=10.0, type=float,
                    help="SK lambda parameter")
parser.add_argument("--nopts", default=400, type=int,
                    help="number of SK opts")

# Queue
parser.add_argument("--max_queue_len", default=0, type=int,
                    help="Maximum number of batches in queue")
parser.add_argument("--queue_start_epoch", default=10, type=int,
                    help="Queue will start at this epoch, if enabled")

# optimization
parser.add_argument("--lr", default=0.01, type=float,
                    help="learning rate")
parser.add_argument("--momentum", default=0.9, type=float,
                    help="sgd momentum")
parser.add_argument("--epochs", default=400, type=int,
                    help="number of epochs to train")
parser.add_argument("--batch-size", default=1024, type=int, metavar="BS",
                    help="batch size")

# logging saving etc.
parser.add_argument("--save_model", action="store_true",
                    help="Save model during training")
parser.add_argument("--datadir", default="./data", type=str,
                    help="datadir")
parser.add_argument("--exp", default="./expe", type=str,
                    help="experimentdir")
parser.add_argument("--type", default="10", type=int,
                    help="cifar10 or 100")

args = parser.parse_args()

# setup_runtime(2, [args.device])
device = "cuda" if torch.cuda.is_available() else "cpu"
knn_dim = 4096
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
logging.info("==> Preparing data..")
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
    cifar_instance = CIFAR10Instance
else:
    cifar_instance = CIFAR100Instance

trainset = cifar_instance(
    root=args.datadir,
    train=True,
    download=True,
    transform=transform_train
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True
)

testset = cifar_instance(
    root=args.datadir,
    train=False,
    download=True,
    transform=transform_test
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=2
)

logging.info("==> Building model..")

numc = [args.ncl] * args.hc
model = models.__dict__[args.arch](num_classes=numc)
knn_dim = 4096

N = len(trainloader.dataset)

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


optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=5e-4
)

model.to(device)

name = "%s" % args.exp.replace("/", "_")
writer = SummaryWriter(f"./runs/cifar{args.type}/{name}")
writer.add_text("args", " \n".join(
    ["%s %s" % (arg, getattr(args, arg)) for arg in vars(args)]))

logging.info(name)


# Training
def train(epoch, SV):
    logging.info("\nEpoch: %d" % epoch)

    # adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        data_time.update(time.time() - end)

        inputs, targets, indexes = (
            inputs.to(device),
            targets.to(device),
            indexes.to(device)
        )

        optimizer.zero_grad()

        outputs = model(inputs)

        if args.hc == 1:
            loss = SV(-outputs)
        else:
            loss = torch.mean(
                torch.stack(
                    [SV(-outputs[h]) for h in range(args.hc)]
                )
            )

        logging.debug(
            "Batch {0} (Size={1}): Loss={2:.5f}".format(
                batch_idx,
                inputs.shape[0],
                loss.item()
            )
        )

        # Use the queue
        if SV.max_n_batches_in_queue > 0:
            if epoch > args.queue_start_epoch:
                logging.debug("Updating the queue")
                SV.update_queue(-outputs)

                if not SV.queue_is_full:
                    logging.debug("Queue is not full yet")
                    continue

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 10 == 0:
            logging.info(
                "Epoch: [{}][{}/{}]"
                "Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Data: {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})"
                .format(
                      epoch,
                      batch_idx,
                      len(trainloader),
                      batch_time=batch_time,
                      data_time=data_time,
                      train_loss=train_loss
                    )
                )
            writer.add_scalar(
                "loss",
                loss.item(),
                batch_idx * 512 + epoch*len(trainloader.dataset)
            )


SV = SinkhornValue(
    epsilon=1./args.lamb,
    solver=pot_sinkhorn,
    max_n_batches_in_queue=args.max_queue_len,
    stopThr=1e-02,
    method="sinkhorn_log",
    numItermax=args.nopts
)

logging.info(SV)

for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch, SV)
    feature_return_switch(model, True)
    acc = kNN(model, trainloader, testloader, K=10, sigma=0.1, dim=knn_dim)
    feature_return_switch(model, False)
    writer.add_scalar("accuracy kNN", acc, epoch)

    if not args.save_model:
        logging.debug("Skipping save model")
        # don"t save model and continue training
        continue

    if acc > best_acc:
        logging.info("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "opt": optimizer.state_dict(),
            "L": selflabels,
        }
        if not os.path.isdir(args.exp):
            os.mkdir(args.exp)
        torch.save(state, "%s/best_ckpt.t7" % (args.exp))
        best_acc = acc
    if epoch % 100 == 0:
        logging.info("Saving..")
        state = {
            "net": model.state_dict(),
            "opt": optimizer.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "L": selflabels,
        }
        if not os.path.isdir(args.exp):
            os.mkdir(args.exp)
        torch.save(state, "%s/ep%s.t7" % (args.exp, epoch))
    if epoch % 50 == 0:
        feature_return_switch(model, True)
        acc = kNN(model, trainloader, testloader, K=[50, 10],
                  sigma=[0.1, 0.5], dim=knn_dim, use_pca=True)
        i = 0
        for num_nn in [50, 10]:
            for sig in [0.1, 0.5]:
                writer.add_scalar("knn%s-%s" % (num_nn, sig), acc[i], epoch)
                i += 1
        feature_return_switch(model, False)
    logging.info("best accuracy: {:.2f}".format(best_acc * 100))

checkpoint = torch.load("%s" % args.exp+"/best_ckpt.t7")
model.load_state_dict(checkpoint["net"])
feature_return_switch(model, True)
acc = kNN(model, trainloader, testloader, K=10,
          sigma=0.1, dim=knn_dim, use_pca=True)
