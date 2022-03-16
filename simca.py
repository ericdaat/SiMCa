import string
import argparse

import torch
import numpy as np
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from simca.dataset import ToyDataset
from simca.model import SiMCa
from simca.sinkhorn import SinkhornLoss, sinkhorn
from simca.utils import assign_with_lap
from simca.viz import plot_distances, plot_embeddings


parser = argparse.ArgumentParser(
    description="SiMCa: Sinkhorn Matrix Factorization with Capacity Constraints"
)

#####################
# Dataset variables #
#####################

parser.add_argument("--n_centers", default=3, type=int,
                    help="Gaussian mixture model with k clusters")

parser.add_argument("--n_items", default=3, type=int,
                    help="Number of items")

parser.add_argument("--n_users", default=1000, type=int,
                    help="Number of users")

parser.add_argument("--distance_weight", default=.3, type=float,
                    help="Distance vs. affinity relative weight")

parser.add_argument("--n_features", default=2, type=int,
                    help="Users and items embedding dimension")

###################
# model variables #
###################

parser.add_argument("--sinkhorn_n_iter", default=400, type=int,
                    help="Number Sinkhorn iterations")

parser.add_argument("--sinkhorn_epsilon", default=.1, type=float,
                    help="Sinkhorn entropy regularization")

parser.add_argument("--n_epochs", default=200, type=int,
                    help="Number of training epochs")

parser.add_argument("--learning_rate", default=5e-3, type=float,
                    help="Learning rate")

args = parser.parse_args()


def main():
    # sample dataset
    toy = ToyDataset(
        n_centers=args.n_centers,
        n_items=args.n_items,
        n_users=args.n_users,
        n_features=args.n_features,
        distance_weight=args.distance_weight
    )

    # declare model
    model = SiMCa(
        capacities=toy.items_capacities,
        alpha=args.distance_weight,
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
        n_iter=args.sinkhorn_n_iter,
        epsilon=args.sinkhorn_epsilon
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        lr=args.learning_rate,
        params=model.parameters()
    )

    # Train model
    training_results = train_model(
        model=model,
        optimizer=optimizer,
        sinkhorn_loss=sinkhorn_loss,
        n_epochs=args.n_epochs,
        toy=toy,
        assigned_item_for_user=toy.assigned_item_for_user,
        assign="lap"
    )

    fig = plot_results(model, toy, training_results)
    fig.savefig("figures/training.png")


def train_model(model, optimizer, sinkhorn_loss, n_epochs, toy,
                assigned_item_for_user, assign="lap"):
    training_results = []

    for epoch_number in range(1, n_epochs+1):
        model.train()
        train_epoch_loss = 0
        optimizer.zero_grad()

        # model out
        M = model(
            toy.users_tensor,
            toy.items_tensor,
            toy.D_tensor
        )

        # loss
        target = torch.nn.functional.one_hot(
            toy.y_true_tensor,
            num_classes=toy.n_items
        )
        loss = sinkhorn_loss(M, target)
        train_epoch_loss += loss.item()
        loss.backward()

        # optimizer
        optimizer.step()

        # Predictions
        model.eval()
        with torch.no_grad():
            a = torch.ones(toy.n_users)
            b = torch.FloatTensor(toy.items_capacities).view(-1)

            P = sinkhorn(
                M,
                a,
                b,
                sinkhorn_loss.epsilon,
                sinkhorn_loss.solver_options["n_iter"]
            )

        if assign == "lap":
            y_pred = assign_with_lap(
                P,
                toy.items_capacities
            )
        elif assign == "argmax":
            y_pred = torch.argmax(P, axis=1)
        else:
            raise NotImplementedError("Invalid assignment method")

        # F1-score evaluation
        f1 = f1_score(
            y_true=assigned_item_for_user,
            y_pred=y_pred,
            average="macro"
        )

        # Embedding distance evaluation
        average_distance = np.linalg.norm(
            model.item_embeddings.weight.data.numpy() - toy.items_features,
            axis=1
        ).mean()

        if epoch_number % 20 == 0:
            print("Epoch {0}: loss={1:.3f}, f1={2:.3f}, avg_dist={3:.3f}"
                  .format(epoch_number, loss.item(), f1, average_distance))

        training_results.append([
            epoch_number,
            loss.item(),
            f1,
            average_distance
        ])

    return training_results


def plot_results(model, toy, training_results):
    training_results_df = pd.DataFrame(
        training_results,
        columns=["epoch", "loss", "F1", "avg_distance"]
    )

    # plot dataset
    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=3)

    # Embeddings space
    ax = plot_embeddings(
        users_features=model.user_embeddings.weight,
        items_features=model.item_embeddings.weight,
        items_capacities=toy.items_capacities,
        y_pred=toy.assigned_item_for_user,
        ax=axs[0]
    )

    # F1 score
    ax = sns.lineplot(
        data=training_results_df,
        x="epoch",
        y="F1",
        ax=axs[1]
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score"),
    ax.set_ylim(0, 1)

    # Embedding distance
    ax = sns.lineplot(
        data=training_results_df,
        x="epoch",
        y="avg_distance",
        ax=axs[2]
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average distance")
    ax.set_ylim(0, None)

    # Annotate with letters
    for i in range(3):
        axs[i].set_anchor("N")

        axs[i].text(
            -0.1,
            1.1,
            string.ascii_uppercase[i],
            transform=axs[i].transAxes,
            size=20,
            weight="bold"
    )

    return fig


if __name__ == "__main__":
    main()
