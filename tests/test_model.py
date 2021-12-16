import torch
import pytest
import pandas as pd

from src.dataset.toy import ToyDataset
from src.ml.model import SiMaC
from src.ml.train import train_model


@pytest.mark.parametrize("epsilon", [0])
@pytest.mark.parametrize("alpha", [0])
@pytest.mark.parametrize("n_items", [3])
@pytest.mark.parametrize("n_users", [1000])
@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0.5])
def test_factory(epsilon, alpha, n_items, n_users, n_features,
                 n_centers, distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    model = SiMaC(
        capacities=toy.items_capacities,
        n_users=n_users,
        epsilon=epsilon,
        alpha=alpha,
        n_features=n_features
    )

    assert isinstance(model, SiMaC)
    assert model.user_embeddings.weight.shape == torch.Size([n_users, 2])
    assert model.item_embeddings.weight.shape == torch.Size([n_items, 2])


@pytest.mark.parametrize("epsilon", [0.1])
@pytest.mark.parametrize("alpha", [0])
@pytest.mark.parametrize("n_items", [3])
@pytest.mark.parametrize("n_users", [1000])
@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_features", [2, 3])
@pytest.mark.parametrize("distance_weight", [0.5])
def test_train(epsilon, alpha, n_items, n_users, n_centers,
               n_features, distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    # train the model
    (y_pred, model, losses_df, scores_df, capacities_df) = train_model(
        toy.users_tensor,
        toy.items_tensor,
        toy.D_tensor,
        toy.y_true_tensor,
        toy.items_capacities,
        toy.items_features,
        n_features=n_features,
        lr=0.01,
        epsilon=epsilon,
        n_iter=10,
        alpha=alpha,
        n_epochs=10,
        users_features=toy.users_features,
        train_user_embeddings=False,
        assign="lap"
    )

    assert y_pred.shape == torch.Size([n_users])

    assert isinstance(model, SiMaC)

    assert isinstance(losses_df, pd.DataFrame)
    assert losses_df.columns.tolist() == ["epoch", "loss"]

    assert isinstance(scores_df, pd.DataFrame)
    assert scores_df.columns.tolist() == ["epoch", "acc", "f1", "distances"]

    assert isinstance(capacities_df, pd.DataFrame)
    assert capacities_df.columns.tolist() == ["center_id", "capacities",
                                              "expected_usage", "actual_usage"]
