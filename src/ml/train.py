import torch
import numpy as np
import scipy
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.ml.model import SiMaC


def assign_with_lap(r_ij, items_capacities):
    """Perform the final assignation with LAP, using
    the allocation matrix as a cost function.

    Args:
        r_ij (torch.FloatTensor): allocation matrix
        items_capacities (torch.FloatTensor): items capacities

    Returns:
        torch.LongTensor: LAP assignations
    """
    r_ij_with_capacities = np.hstack(
        [np.repeat(r_ij.data.numpy()[:, i].reshape(-1, 1), c, axis=1)
        for i, c in enumerate(items_capacities[0])]
    )

    column_to_item_ix = np.hstack(
        [np.repeat([i], c)
        for i, c in enumerate(items_capacities[0])]
    )

    _, col_ix = scipy.optimize.linear_sum_assignment(
        r_ij_with_capacities,
        maximize=True
    )

    y_pred = np.asarray([column_to_item_ix[i] for i in col_ix])

    return torch.from_numpy(y_pred)


def format_training_results_in_dataframes(y_pred, capacities, e_usage,
                                          losses, scores):
    """Summarise the training results in DataFrames, used for data viz.

    Args:
        y_pred (torch.LongTensor): model assignations
        capacities (torch.LongTensor): items capacities
        e_usage (torch.FloatTensor): items expected usage
        losses (list): loss function values per epoch
        scores (list): scores values per epoch

    Returns:
        tuple: capacities, losses and scores dataframes
    """
    capacities_df = pd.DataFrame(
        dict(
            center_id=["Item {0}".format(i)
                       for i in np.arange(1, capacities.shape[1]+1)],
            capacities=capacities.reshape(-1),
            expected_usage=e_usage.data.numpy(),
            actual_usage=torch.nn.functional.one_hot(
                y_pred,
                num_classes=capacities.shape[1]
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
            f1=[s[1] for s in scores],
            distances=[s[2] for s in scores]
        )
    )

    return capacities_df, losses_df, scores_df


def train_model(users_tensor, items_tensor, D_tensor, y_true_tensor, items_capacities,
                items_features, lr, epsilon, n_iter, alpha, n_epochs, n_features,
                users_features=None, train_user_embeddings=False, assign="lap"):
    """Define and train the model.

    Args:
        users_tensor (torch.LongTensor): users ids
        items_tensor (torch.LongTensor): items ids
        D_tensor (torch.FloatTensor): travel distance matrix
        y_true_tensor (torch.LongTensor): observed allocations
        items_capacities (torch.FloatTensor): items capacities
        items_features (torch.FloatTensor): items features
        lr (float): Learning rate
        epsilon (float]: Entropy regularization parameter
        n_iter (int): Number of Sinkhorn iterations
        alpha (float): Distance weight, 0 being no distance involved.
        n_epochs (int): Number of epochs
        n_features (int): Number of latent features for items and users
        users_features (np.ndarray, optional): Users features. Defaults to None.
        train_user_embeddings (bool, optional): Make the users embeddings trainable. Defaults to False.
        assign (str, optional): Assignment method, can be LAP or argmax. Defaults to "lap".

    Returns:
        tuple: (y_pred, model, losses_df, scores_df, capacities_df)
    """

    if users_features is not None:
        users_features = torch.FloatTensor(users_features)

    model = SiMaC(
        capacities=items_capacities,
        epsilon=epsilon,
        alpha=alpha,
        n_iter=n_iter,
        n_users=users_tensor.shape[0],
        n_features=n_features,
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
        r_ij = model(users_tensor, items_tensor, D_tensor)
        e_usage = r_ij.sum(axis=0)

        # predictions
        if assign=="argmax":
            y_pred = torch.argmax(r_ij, axis=1)
        else:
            y_pred = assign_with_lap(r_ij, items_capacities)

        # loss function
        nll_loss = criterion(torch.log(r_ij), y_true_tensor)

        loss = nll_loss
        train_epoch_loss += loss.item()
        loss.backward()

        # optimizer
        optimizer.step()

        # training stats
        losses.append([loss.item()])
        acc = accuracy_score(y_true=y_true_tensor, y_pred=y_pred)
        f1 = f1_score(y_true=y_true_tensor, y_pred=y_pred, average="macro")

        embedding_distance = np.linalg.norm(
            model.item_embeddings.weight.data.numpy()
            - items_features,
            axis=1
        )

        scores.append([acc, f1, np.mean(embedding_distance)])

    ##########
    # Scores #
    ##########
    capacities_df, losses_df, scores_df = \
        format_training_results_in_dataframes(
            y_pred, items_capacities, e_usage,
            losses, scores
        )

    return (y_pred, model,
            losses_df, scores_df, capacities_df)
