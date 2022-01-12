import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from src.ml.sinkhorn import SinkhornValue, sinkhorn

batch_size=64


class CIFAR10Instance(torchvision.datasets.CIFAR10):
    """
    https://github.com/yukimasano/self-label/blob/581957c2fcb3f14a0382cf71a3d36b21b9943798/cifar_utils.py#L5
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(CIFAR10Instance, self).__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )


    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image)

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def main():
    # Load CIFAR-10
    transform_train = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
            transforms.ToTensor(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

    trainset = CIFAR10Instance(
        root="../data/cifar-10",
        train=True,
        download=True,
        transform=transform_train
    )
    testset = CIFAR10Instance(
        root="../data/cifar-10",
        train=False,
        download=True,
        transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=100,
        shuffle=True,
        num_workers=0
    )

    classes = (
        "plane", "car", "bird", "cat",
        "deer", "dog", "frog", "horse",
        "ship", "truck"
    )

    N = len(trainloader.dataset)
    K = 128  # number of clusters

    selflabels = np.zeros(N, dtype=np.int32)

    for qq in range(N):
        selflabels[qq] = qq % K

    selflabels = np.random.permutation(selflabels)
    selflabels = torch.LongTensor(selflabels)


    # Load Alexnet model, with output size = K (128)
    model = torchvision.models.alexnet(pretrained=False, num_classes=K)

    # marginals
    a = torch.ones(batch_size) / batch_size  # minibatch size
    b = torch.ones(K) / K                    # K clusters (128)

    # ADAM optimizer
    optimizer = torch.optim.SGD(lr=0.01, params=model.parameters())

    for epoch in range(10):
        epoch_loss = 0

        # loop over minibatches
        for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
            # train mode
            model.train()

            # set gradients to zero
            optimizer.zero_grad()

            # compute inputs (images) representations
            x = model(inputs)
            P = torch.nn.LogSoftmax(dim=1)(x)

            # compute softmax probabilities over each cluster ()
            M = P - np.log(inputs.shape[0])

            # init Sinkhorn loss
            SV = SinkhornValue(
                a,
                b,
                epsilon=0.1,
                solver=sinkhorn,
                n_iter=10
            )

            # compute Sinkhorn loss
            loss = -SV(M)

            # compute gradients
            loss.backward()

            # backpropagation
            optimizer.step()

            epoch_loss += loss.item()

        print(epoch_loss / (batch_idx+1))


if __name__ == "__main__":
    main()
