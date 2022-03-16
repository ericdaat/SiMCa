import torch
import pytest

from simca.dataset import ToyDataset
from simca.model import SiMCa
from simca.train import train_model
from simca.sinkhorn import SinkhornLoss, sinkhorn


@pytest.mark.parametrize("n_centers", [3, 6])
@pytest.mark.parametrize("n_items", [3, 6])
@pytest.mark.parametrize("n_users", [100])
@pytest.mark.parametrize("n_features", [2, 4])
@pytest.mark.parametrize("distance_weight", [0, .3])
@pytest.mark.parametrize("n_epochs", [10])
@pytest.mark.parametrize("sinkhorn_n_iter", [10, 100])
@pytest.mark.parametrize("sinkhorn_epsilon", [.1, .5])
@pytest.mark.parametrize("learning_rate", [5e-3])
def test_train_model(n_centers, n_items, n_users, n_features,
                     distance_weight,
                     sinkhorn_n_iter, sinkhorn_epsilon,
                     n_epochs, learning_rate):
    # sample dataset
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    # declare model
    model = SiMCa(
        capacities=toy.items_capacities,
        alpha=distance_weight,
        n_users=toy.n_users,
        n_features=toy.n_features,
        user_embeddings=torch.FloatTensor(toy.users_features),
        train_user_embeddings=False
    )

    # Sinkhorn loss
    a = torch.ones(toy.n_users)
    b = torch.FloatTensor(toy.items_capacities).view(-1)

    sinkhorn_loss = SinkhornLoss(
        a=a,
        b=b,
        solver=sinkhorn,
        n_iter=sinkhorn_n_iter,
        epsilon=sinkhorn_epsilon
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        lr=learning_rate,
        params=model.parameters()
    )

    # Train model
    training_results = train_model(
        model=model,
        optimizer=optimizer,
        sinkhorn_loss=sinkhorn_loss,
        n_epochs=n_epochs,
        toy=toy,
        assigned_item_for_user=toy.assigned_item_for_user,
        assign="lap"
    )
