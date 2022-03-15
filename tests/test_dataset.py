import os

import torch
import numpy as np
import pandas as pd
import pytest

from simca.dataset import ToyDataset, save_dataset, load_dataset


n_centers = 3
n_items = 3
n_users = 10
distance_weight = 0


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_items", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
def test_factory(n_centers, n_items, n_users, n_features,
                 distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assert isinstance(toy, ToyDataset)
    assert toy.n_centers == n_centers
    assert toy.n_items == n_items
    assert toy.n_users == n_users
    assert toy.distance_weight == distance_weight
    assert toy.users_features.shape == (n_users, n_features)
    assert toy.items_features.shape == (n_items, n_features)


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_items", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
def test_capacities(n_centers, n_items, n_users, n_features,
                    distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assert type(toy.items_capacities) == np.ndarray
    assert toy.items_capacities.shape == (1, n_items)
    assert np.min(toy.items_capacities) > 0
    assert np.sum(toy.items_capacities) > n_users


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_items", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
def test_users(n_centers, n_items, n_users, n_features,
               distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assert type(toy.users_features) == np.ndarray
    assert toy.users_features.shape == (n_users, 2)


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_items", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
def test_distance(n_centers, n_items, n_users, n_features,
                  distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assert type(toy.D) == np.ndarray
    assert toy.D.shape == (n_users, n_items)
    assert np.min(toy.D) >= 0


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_items", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
def test_assignment(n_centers, n_items, n_users, n_features,
                    distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assert type(toy.assigned_item_for_user) == np.ndarray
    assert toy.assigned_item_for_user.shape == (n_users,)
    assert all(np.unique(toy.assigned_item_for_user, return_counts=True)[1] \
                <= toy.items_capacities[0])


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_items", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0])
@pytest.mark.parametrize("n_permutations", [0, 10, 20])
def test_permutations(n_centers, n_items, n_users, n_features,
                      distance_weight, n_permutations):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assignations_before = toy.assigned_item_for_user
    assignations_noisy = toy.add_random_permutations(n_permutations)

    # check that the capacities are the same
    assert pd.Series(assignations_before).value_counts().tolist() \
        == pd.Series(assignations_noisy).value_counts().tolist()

    if n_permutations > 0:
        # check that there is at least one difference between arrays
        assert any(assignations_before != assignations_noisy)
    else:
        assert any(assignations_before == assignations_noisy)



@pytest.mark.parametrize("n_items", [3])
@pytest.mark.parametrize("n_users", [1000])
@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_features", [2])
@pytest.mark.parametrize("distance_weight", [0.5])
def test_tensors(n_items, n_users, n_centers, n_features,
                 distance_weight):
    toy = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=n_features,
        distance_weight=distance_weight
    )

    assert toy.users_tensor.shape == torch.Size([n_users,])
    assert toy.items_tensor.shape == torch.Size([n_users, n_items])
    assert toy.D_tensor.shape == torch.Size([n_users, n_items])
    assert toy.y_true_tensor.shape == torch.Size([n_users])

    assert isinstance(toy.users_tensor, torch.LongTensor)
    assert isinstance(toy.items_tensor, torch.LongTensor)
    assert isinstance(toy.D_tensor, torch.FloatTensor)
    assert isinstance(toy.y_true_tensor, torch.LongTensor)


def test_save_and_load():
    toy_1 = ToyDataset(
        n_centers=n_centers,
        n_items=n_items,
        n_users=n_users,
        n_features=2,
        distance_weight=distance_weight
    )

    save_path = "toy.pkl"
    save_dataset(toy_1, save_path)
    toy_2 = load_dataset(save_path)

    assert all(toy_1.assigned_item_for_user == toy_2.assigned_item_for_user)

    os.remove(save_path)
