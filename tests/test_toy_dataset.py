import os

import numpy as np
import pandas as pd
import pytest

from src.dataset.toy import ToyDataset, save_dataset, load_dataset


n_centers = 3
n_pois = 3
n_users = 10
distance_weight = 0


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_pois", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
@pytest.mark.parametrize("noise", [0])
def test_factory(n_centers, n_pois, n_users, distance_weight, noise):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=noise
    )

    assert isinstance(toy, ToyDataset)
    assert toy.n_centers == n_centers
    assert toy.n_pois == n_pois
    assert toy.n_users == n_users
    assert toy.distance_weight == distance_weight


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_pois", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
@pytest.mark.parametrize("noise", [0])
def test_capacities(n_centers, n_pois, n_users, distance_weight, noise):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=noise
    )

    assert type(toy.pois_capacities) == np.ndarray
    assert toy.pois_capacities.shape == (1, n_pois)
    assert np.min(toy.pois_capacities) > 0
    assert np.sum(toy.pois_capacities) > n_users


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_pois", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
@pytest.mark.parametrize("noise", [0])
def test_users(n_centers, n_pois, n_users, distance_weight, noise):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=noise
    )

    assert type(toy.users_features) == np.ndarray
    assert toy.users_features.shape == (n_users, 2)


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_pois", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
@pytest.mark.parametrize("noise", [0])
def test_distance(n_centers, n_pois, n_users, distance_weight, noise):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=noise
    )

    assert type(toy.D) == np.ndarray
    assert toy.D.shape == (n_users, n_pois)
    assert np.min(toy.D) >= 0


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_pois", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("distance_weight", [0, 0.5, 1])
@pytest.mark.parametrize("noise", [0])
def test_assignment(n_centers, n_pois, n_users, distance_weight, noise):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=noise
    )

    assert type(toy.assigned_poi_for_user) == np.ndarray
    assert toy.assigned_poi_for_user.shape == (n_users,)
    assert all(np.unique(toy.assigned_poi_for_user, return_counts=True)[1] \
                <= toy.pois_capacities[0])


@pytest.mark.parametrize("n_centers", [3])
@pytest.mark.parametrize("n_pois", [3, 4, 5])
@pytest.mark.parametrize("n_users", [100, 200])
@pytest.mark.parametrize("distance_weight", [0])
@pytest.mark.parametrize("noise", [0, .1, .5, 1])
def test_permutations(n_centers, n_pois, n_users, distance_weight, noise):
    toy = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=0
    )

    assignations_before = toy.assigned_poi_for_user
    toy.noise_dataset(noise)
    assignations_noisy = toy.assigned_poi_for_user

    # check that the capacities are the same
    assert pd.Series(assignations_before).value_counts().tolist() \
        == pd.Series(assignations_noisy).value_counts().tolist()

    if noise > 0:
        # check that there is at least one difference between arrays
        assert any(assignations_before != assignations_noisy)
    else:
        assert any(assignations_before == assignations_noisy)


def test_save_and_load():
    toy_1 = ToyDataset(
        n_centers=n_centers,
        n_pois=n_pois,
        n_users=n_users,
        distance_weight=distance_weight,
        noise=0
    )

    save_path = "toy.pkl"
    save_dataset(toy_1, save_path)
    toy_2 = load_dataset(save_path)

    assert all(toy_1.assigned_poi_for_user == toy_2.assigned_poi_for_user)

    os.remove(save_path)
