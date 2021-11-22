import torch
import numpy as np
import scipy
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.ml.model import ModelOT


def prepare_input_data(toy):
    users_tensor = torch.from_numpy(np.arange(0, toy.n_users))

    pois_tensor = torch.from_numpy(
        np.repeat(np.arange(toy.pois_capacities.shape[1]).reshape(1, -1),
                toy.users_features.shape[0],
                axis=0)
    ).long()

    D_tensor = torch.FloatTensor(toy.D)

    y_true = torch.from_numpy(toy.assigned_poi_for_user).long()

    return users_tensor, pois_tensor, D_tensor, y_true


def assign_with_lap(r_ij, pois_capacities):
    r_ij_with_capacities = np.hstack(
        [np.repeat(r_ij.data.numpy()[:, i].reshape(-1, 1), c, axis=1)
        for i, c in enumerate(pois_capacities[0])]
    )

    column_to_poi_ix = np.hstack(
        [np.repeat([i], c)
        for i, c in enumerate(pois_capacities[0])]
    )

    _, col_ix = scipy.optimize.linear_sum_assignment(
        r_ij_with_capacities,
        maximize=True
    )

    y_pred = np.asarray([column_to_poi_ix[i] for i in col_ix])

    return torch.from_numpy(y_pred)


def train_model(users_tensor, pois_tensor, D_tensor, y_true, pois_capacities,
                lr, epsilon, n_iter, alpha, n_epochs,
                users_features=None, train_user_embeddings=False, assign="lap"):

    if users_features is not None:
        users_features = torch.FloatTensor(users_features)

    model = ModelOT(
        capacities=pois_capacities,
        epsilon=epsilon,
        alpha=alpha,
        n_iter=n_iter,
        n_users=users_tensor.shape[0],
        user_embeddings=users_features,
        train_user_embeddings=train_user_embeddings
    )

    model.train()

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    criterion = torch.nn.NLLLoss()

    losses = []
    scores = []

    #########
    # Train #
    #########

    for _ in range(n_epochs):
        train_epoch_loss = 0

        optimizer.zero_grad()

        # model out
        r_ij = model(users_tensor, pois_tensor, D_tensor)
        e_usage = r_ij.sum(axis=0)

        # predictions
        if assign=="sample":
            y_pred = torch.distributions.Categorical(torch.exp(r_ij)).sample()
        elif assign=="argmax":
            y_pred = torch.argmax(r_ij, axis=1)
        else:
            y_pred = assign_with_lap(r_ij, pois_capacities)

        # loss function
        nll_loss = criterion(torch.log(r_ij), y_true)

        # not loss anymore we maximize it!
        capacity_loss = torch.sum(model.affinity_matrix*r_ij) / users_tensor.shape[0]

        loss = nll_loss
        train_epoch_loss += loss.item()
        loss.backward()

        # optimizer
        optimizer.step()

        # training stats
        losses.append([loss.item(), capacity_loss.item()])
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        scores.append([acc, f1])

    ##########
    # Scores #
    ##########

    capacities_df = pd.DataFrame(
        dict(
            center_id=["POI {0}".format(i) for i in np.arange(1, pois_capacities.shape[1]+1)],
            capacities=pois_capacities.reshape(-1),
            expected_usage=e_usage.data.numpy(),
            actual_usage=torch.nn.functional.one_hot(
                y_pred,
                num_classes=pois_capacities.shape[1]
            ).sum(axis=0).data.numpy()
        )
    )

    losses_df = pd.DataFrame(
        dict(
            epoch=["{0}".format(i) for i in np.arange(1, len(losses)+1)],
            loss=[l[0] for l in losses]
        )
    )

    scores_df = pd.DataFrame(
        dict(
            epoch=["{0}".format(i) for i in np.arange(1, len(losses)+1)],
            acc=[s[0] for s in scores],
            f1=[s[1] for s in scores]
        )
    )

    return (y_pred, model,
            losses_df, scores_df, capacities_df)
