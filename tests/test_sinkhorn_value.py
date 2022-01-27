import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset.blobs_clusters import BlobsDataset
from src.ml.sinkhorn import SinkhornValue, pot_sinkhorn


dataset = BlobsDataset(
    n_features=512,
    n_clusters=10,
    n_samples=2000
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    drop_last=True
)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, inputs, return_features=False):
        mlp_out = self.mlp(inputs)
        softmax_out = nn.LogSoftmax(dim=1)(mlp_out)

        if return_features:
            return mlp_out
        else:
            return softmax_out


def test_forward():
    sinkhorn_value = SinkhornValue(
        epsilon=.2,
        solver=pot_sinkhorn,
        max_n_batches_in_queue=2,
        method="sinkhorn_log",
        numItermax=100,
        warn=True
    )

    inputs, _ = next(iter(dataloader))
    loss = sinkhorn_value(inputs)

    assert isinstance(loss, torch.FloatTensor)


def test_queue_update():
    sinkhorn_value = SinkhornValue(
        epsilon=.2,
        solver=pot_sinkhorn,
        max_n_batches_in_queue=2,
        method="sinkhorn_log",
        numItermax=100,
        warn=True
    )

    zeros = torch.zeros([32, 128])
    ones = torch.ones([32, 128])

    # first update: append, no roll
    sinkhorn_value.update_queue(zeros)
    assert sinkhorn_value.stored_M.shape[0] == 32
    assert torch.all(sinkhorn_value.stored_M[:32, :] == zeros)

    # second update: append, still no roll
    sinkhorn_value.update_queue(zeros)
    assert sinkhorn_value.stored_M.shape[0] == 64
    assert torch.all(sinkhorn_value.stored_M[:32, :] == zeros)
    assert torch.all(sinkhorn_value.stored_M[32:64, :] == zeros)

    # third update: roll, last batch comes first
    sinkhorn_value.update_queue(ones)
    assert sinkhorn_value.stored_M.shape[0] == 64
    assert sinkhorn_value.stored_M.max() == 1
    assert torch.all(sinkhorn_value.stored_M[:32, :] == ones)
    assert torch.all(sinkhorn_value.stored_M[32:64, :] == zeros)
