import torch
import pytest

from src.dataset.toy import ToyDataset
from src.ml.model import ModelOT
from src.ml.train import train_model, prepare_input_data


@pytest.mark.parametrize("epsilon", [0])
@pytest.mark.parametrize("alpha", [0])
@pytest.mark.parametrize("n_pois", [3])
@pytest.mark.parametrize("n_users", [1000])
@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("distance_weight", [0.5])
def test_factory(epsilon, alpha, n_pois, n_users, n_centers, distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=0
    )

    model = ModelOT(
        capacities=toy.pois_capacities,
        n_users=n_users,
        epsilon=epsilon,
        alpha=alpha
    )

    assert isinstance(model, ModelOT)
    assert model.user_embeddings.weight.shape == torch.Size([n_users, 2])
    assert model.poi_embeddings.weight.shape == torch.Size([n_pois, 2])


@pytest.mark.parametrize("n_pois", [3])
@pytest.mark.parametrize("n_users", [1000])
@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("distance_weight", [0.5])
def test_input_data(n_pois, n_users, n_centers, distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=0
    )

    users_tensor, pois_tensor, D_tensor, y_true = prepare_input_data(toy)

    assert users_tensor.shape == torch.Size([n_users,])
    assert pois_tensor.shape == torch.Size([n_users, n_pois])
    assert D_tensor.shape == torch.Size([n_users, n_pois])
    assert y_true.shape == torch.Size([n_users])

    assert isinstance(users_tensor, torch.LongTensor)
    assert isinstance(pois_tensor, torch.LongTensor)
    assert isinstance(D_tensor, torch.FloatTensor)
    assert isinstance(y_true, torch.LongTensor)


@pytest.mark.parametrize("epsilon", [0])
@pytest.mark.parametrize("alpha", [0])
@pytest.mark.parametrize("n_pois", [3])
@pytest.mark.parametrize("n_users", [1000])
@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("distance_weight", [0.5])
def test_train(epsilon, alpha, n_pois, n_users, n_centers, distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=0
    )

    # prepare input
    users_tensor, pois_tensor, D_tensor, y_true = prepare_input_data(toy)

    # train the model
    (y_pred, model, _, _, _) = train_model(
        users_tensor,
        pois_tensor,
        D_tensor,
        y_true,
        toy.pois_capacities,
        lr=0.01,
        epsilon=0.1,
        n_iter=10,
        alpha=0,
        n_epochs=10,
        users_features=toy.users_features,
        train_user_embeddings=False
    )

    assert y_pred.shape == torch.Size([n_users])
