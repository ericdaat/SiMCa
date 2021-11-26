import math
import pickle

import numpy as np
import scipy
import torch
from sklearn.datasets import make_blobs

from src.utils import compute_distance


CIRCLE_RADIUS = 500


def save_dataset(toy, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(toy, f)


def load_dataset(filepath):
    with open(filepath, "rb") as f:
        toy = pickle.load(f)

    return toy

class ToyDataset(object):
    def __init__(self, n_users, n_items, n_centers, n_features,
                 distance_weight):
        """Constructor for the toy dataset.

        Args:
            n_users (int): Number of generated users
            n_items (int): Number of generated points of interest
            n_centers (int): Number of clusters of the gaussian mixture model
            n_features (int): Number of latent dimensions for users and items.
            distance_weight (float): Importance of the distance in the
                assignment phase.
            noise (float): Percentage of noise to add to the dataset.
                This percentage corresponds to the number of users that
                will be permutated with other users after the assignment step.
        """

        # configuration parameters
        self.n_users = n_users
        self.n_items = n_items
        self.n_centers = n_centers
        self.distance_weight = distance_weight

        # initialize dataset variables
        self.users_x = None
        self.users_y = None
        self.items_x = None
        self.items_y = None
        self.items_capacities = None
        self.D = None
        self.users_features = None
        self.items_features = None
        self.assigned_item_for_user = None

        # tensors
        self.users_tensor = None
        self.items_tensor = None
        self.D_tensor = None
        self.y_true_tensor = None

        # generate the dataset
        self.update_dataset(n_users, n_items, n_centers,
                            n_features, distance_weight)

    def __len__(self):
        return self.n_users

    def __getitem__(self, idx):
        return (self.users_tensor[idx], self.items_tensor[idx],
                self.D_tensor[idx], self.y_true_tensor[idx])

    def update_dataset(self, n_users, n_items, n_centers,
                       n_features, distance_weight):
        """Update the dataset by creating new users, new items and new
        assignments.

        Args:
            n_users (int): Number of generated users
            n_items (int): Number of generated points of interest
            n_centers (int): Number of clusters of the gaussian mixture model
            distance_weight (float): Importance of the distance in the
                assignment phase.
        """
        (users_x, users_y, items_x, items_y, items_capacities,
         D, users_features, items_features,
         assigned_item_for_user) =\
                self.__generate(n_users, n_items, n_centers,
                                n_features, distance_weight)

        # Update setup variables
        self.n_users = n_users
        self.n_items = n_items
        self.n_centers = n_centers
        self.distance_weight = distance_weight

        # Update dataset
        self.users_x = users_x
        self.users_y = users_y
        self.items_x = items_x
        self.items_y = items_y
        self.items_capacities = items_capacities
        self.D = D
        self.users_features = users_features
        self.items_features = items_features
        self.assigned_item_for_user = assigned_item_for_user

        # Convert to tensor
        (users_tensor, items_tensor,
         D_tensor, y_true_tensor) = self.__convert_to_tensors()

        self. users_tensor = users_tensor
        self. items_tensor = items_tensor
        self. D_tensor = D_tensor
        self. y_true_tensor = y_true_tensor

    def add_random_permutations(self, n_permutations):
        """Add random permutations to the allocations, by swapping
        users two by two.

        Args:
            n_permutations (int): Number of users to randomly swap
                with another user.

        Returns:
            np.array: new assignations
        """
        new_assigned_item_for_user = self.assigned_item_for_user.copy()

        users_indices = np.random.choice(
            a=range(self.n_users),
            size=n_permutations,
            replace=False
        )

        for user_index in users_indices:
            user_assigned_item = new_assigned_item_for_user[user_index]
            user_new_item = np.random.choice([i for i in range(self.n_items)
                                               if i != user_assigned_item])
            users_from_new_item = np.where(new_assigned_item_for_user == user_new_item)[0]
            user_to_switch_with = np.random.choice(users_from_new_item)

            # apply permutations
            new_assigned_item_for_user[user_index] = user_new_item
            new_assigned_item_for_user[user_to_switch_with] = user_assigned_item

        return new_assigned_item_for_user

    def add_gaussian_noise_to_users_features(self, ratio, normalize=True):
        """Add gaussian noise to the users features.

        Args:
            ratio (float): Relative weight of the Gaussian noise, compared
                to the users features.
            normalize (bool, optional): Normalize the newly obtained features. Defaults to True.

        Returns:
            np.ndarray: Noised users features
        """
        noise = np.random.normal(0, .1, self.users_features.shape)

        users_features = (1-ratio) * self.users_features + ratio * noise

        if normalize:
            users_features /= np.linalg.norm(users_features, axis=1).reshape(-1, 1)

        return users_features

    def __generate(self, n_users, n_items, n_centers,
                   n_features, distance_weight):
        """Generate users, items and assignments.

        Args:
            n_users (int): Number of generated users
            n_items (int): Number of generated points of interest
            n_centers (int): Number of clusters of the gaussian mixture model
            distance_weight (float): Importance of the distance in the
                assignment phase.

        Returns:
            tuple: generated data
        """
        users_x, users_y, items_x, items_y, items_capacities = \
            self.__generate_users_and_items_coordinates(
                n_users,
                n_items
            )

        D = self.__compute_travel_distance(users_x, users_y, items_x, items_y)

        users_features, items_features = self.__generate_users_and_items_features(
            n_users,
            n_items,
            n_centers,
            n_features=n_features,
            normalize=True
        )

        assigned_item_for_user = self.__assign_users_to_items(
            users_features,
            items_features,
            D,
            items_capacities,
            distance_weight
        )

        return (
            users_x, users_y, items_x, items_y, items_capacities,
            D, users_features, items_features,
            assigned_item_for_user
        )


    def __generate_users_and_items_coordinates(self, n_users, n_items):
        """Generate users and items coordinates.

        Args:
            n_users (int): Number of generated users
            n_items (int): Number of generated points of interest

        Returns:
            tuple: Users and items coordinates
        """
        items_capacities = np.round(
            np.multiply(
                n_users,
                np.random.dirichlet(np.ones(n_items), size=1)
            )
        ).astype(int)
        items_capacities = items_capacities + 10  # add extra spots

        items_x = []
        items_y = []
        for i in np.random.rand(n_items):
            alpha = 2 * math.pi * i
            x_i = np.cos(alpha) * CIRCLE_RADIUS
            y_i = np.sin(alpha) * CIRCLE_RADIUS

            items_x.append(x_i)
            items_y.append(y_i)

        # Patients location
        users_x = []
        users_y = []
        for i in np.random.rand(n_users):
            alpha = 2 * math.pi * i
            x_i = np.cos(alpha) * CIRCLE_RADIUS
            y_i = np.sin(alpha) * CIRCLE_RADIUS

            users_x.append(x_i)
            users_y.append(y_i)

        return users_x, users_y, items_x, items_y, items_capacities


    def __compute_travel_distance(self, users_x, users_y, items_x, items_y):
        """Compute the distance matrix between users and items, using
        circle distance.

        Args:
            users_x (np.array): users x coordinates
            users_y (np.array): users y coordinates
            items_x (np.array): items x coordinates
            items_y (np.array): items y coordinates

        Returns:
            np.ndarray: distance matrix
        """
        D = np.zeros(
            shape=(len(users_x), len(items_x))
        )

        for i in range(len(users_x)):
            for j in range(len(items_x)):
                # returns straight_distance, circle_distance
                _, d_ij = compute_distance(
                    users_x[i],
                    users_y[i],
                    items_x[j],
                    items_y[j],
                    radius=CIRCLE_RADIUS
                )

                D[i][j] = d_ij

        return D


    def __generate_users_and_items_features(self, n_users, n_items,
                                           n_centers, n_features,
                                           normalize=True):
        """Generate users and items features.

        Args:
            n_users (int): Number of generated users
            n_items (int): Number of generated points of interest
            n_centers (int): Number of clusters of the gaussian mixture model
            normalize (bool, optional): Normalize features
                (each vector has unit norm). Defaults to True.

        Returns:
            tuple: users and items features
        """
        X, group = make_blobs(
            n_samples=n_users+n_items,
            n_features=n_features,
            centers=n_centers,
            random_state=1
        )

        n_points_per_center = [len(item)
                                for item in np.array_split(np.arange(n_items),
                                                           n_centers)]

        # pick points from each group and convert it to a item
        items_rows = []
        for i in range(n_centers):
            items_rows += list(
                np.random.choice(
                    np.where(group==i)[0],
                    n_points_per_center[i],
                    replace=False
                )
            )

        items_features = X[items_rows, :]
        users_features = np.delete(X, items_rows, axis=0)

        if normalize:
            users_features /= np.linalg.norm(users_features, axis=1).reshape(-1, 1)
            items_features /= np.linalg.norm(items_features, axis=1).reshape(-1, 1)

        return users_features, items_features


    def __assign_users_to_items(self, users_features, items_features, D,
                               items_capacities, distance_weight):
        """Assign users to points of interest

        Args:
            users_features (np.ndarray): users features
            items_features (np.ndarray): points of interest features
            D (np.ndarray): Distance matrix between users and points of interest
            items_capacities (np.ndarray): Capacities of the points of interest
            distance_weight (float): Importance of the distance in the
                assignment phase.

        Returns:
            [type]: [description]
        """
        C = (1 - distance_weight) * np.dot(users_features, items_features.T) \
            - distance_weight * (D / np.mean(D))

        C_with_capacity = np.hstack(
            [np.repeat(C[:, i].reshape(-1, 1), c, axis=1)
            for i, c in enumerate(items_capacities[0])]
        )

        column_to_item_ix = np.hstack(
            [np.repeat([i], c)
            for i, c in enumerate(items_capacities[0])]
        )

        row_ix, col_ix = scipy.optimize.linear_sum_assignment(
            C_with_capacity,
            maximize=True
        )

        assigned_item_for_user = np.asarray(
            [column_to_item_ix[i] for i in col_ix]
        )

        return assigned_item_for_user

    def __convert_to_tensors(self):
        users_tensor = torch.LongTensor(np.arange(0, self.n_users))

        items_tensor = torch.LongTensor(
            np.repeat(
                np.arange(self.items_capacities.shape[1]).reshape(1, -1),
                self.users_features.shape[0],
                axis=0
            )
        )

        D_tensor = torch.FloatTensor(self.D)

        y_true_tensor = torch.LongTensor(self.assigned_item_for_user)

        return users_tensor, items_tensor, D_tensor, y_true_tensor
