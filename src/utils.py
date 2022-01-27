import torch
import numpy as np


def compute_distance(x1, y1, x2, y2, radius):
    # Straight distance
    straight_distance = np.sqrt(
        (y2 - y1)**2 + (x2 - x1)**2
    )

    # Great circle distance
    alpha = np.arctan2(y1, x1) - np.arctan2(y2, x2)
    circle_distance = radius * alpha
    circle_distance = abs(circle_distance)

    return straight_distance, circle_distance


class AverageMeter(object):
    """computes and stores the average and current value"""

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
