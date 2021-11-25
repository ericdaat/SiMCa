import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn

from src.config import CENTERS_COLORS

sns.set(style="whitegrid", palette="Set2")


def plot_capacities(capacities_df, ax):
    ax.cla()
    capacities_df.plot.bar(x="center_id", ax=ax)

    diff = (capacities_df["actual_usage"] - capacities_df["capacities"]).values
    overflow = np.where(diff > 0, diff, 0).sum()
    ax.set_title("Capacities (Overflow={0:.0f})".format(overflow))
    ax.set_xlabel("Item")
    ax.set_ylabel("Number of users")


def plot_losses(losses_df, ax):
    ax.cla()
    losses_df.plot.line(x="epoch", ax=ax)

    ax.set_title("Loss function")
    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Loss")


def plot_scores(scores_df, ax):
    ax.cla()
    scores_df.plot.line(x="epoch", ax=ax)

    ax.set_ylim(0, 1)

    ax.set_title("Scores: acc={0:.3f}; f1={1:.3f}"
                .format(scores_df["acc"].iloc[-1],
                        scores_df["f1"].iloc[-1]))

    ax.set_xlabel("Epoch number")
    ax.set_ylabel("Score")


def plot_embeddings(users_features,
                    items_features,
                    items_capacities,
                    y_pred, ax):
    # Normalize
    items_features =nn.functional.normalize(
        items_features,
        dim=1
    ).data.numpy()

    users_features =nn.functional.normalize(
        users_features,
        dim=1
    ).data.numpy()

    # Plot patients
    if y_pred is not None:
        colors = [CENTERS_COLORS[i] for i in y_pred]
    else:
        colors = "black"

    ax.scatter(
        x=users_features[:, 0],
        y=users_features[:, 1],
        s=30,
        marker="x",
        color=colors,
        alpha=.5
    )

    ax.scatter(
        x=items_features[:, 0],
        y=items_features[:, 1],
        c=CENTERS_COLORS[:items_capacities.shape[1]],
        s=items_capacities,
        alpha=.7,
        marker="o",
        edgecolors="black"
    )

    # Plot xy axis
    ax.axhline(y=0, color="black", linestyle="--", lw=1, alpha=.5)
    ax.axvline(x=0, color="black", linestyle="--", lw=1, alpha=.5)

    # Plot setup
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.set_aspect("equal")
    ax.axis("off")


def plot_distances(users_xy, items_xy, radius, y_pred,
                   items_capacities, ax):
    # Draw base circle
    theta = np.linspace(0, 2*np.pi, 100)
    x1 = np.cos(theta) * radius
    x2 = np.sin(theta) * radius
    ax.plot(x1, x2, linewidth=.5, color="grey")

    # Draw "care centers"
    ax.scatter(
        items_xy[:, 0],
        items_xy[:, 1],
        s=2*items_capacities,
        alpha=.7,
        label="care center",
        c=CENTERS_COLORS[:items_capacities.shape[1]],
        edgecolors="black"
    )

    if y_pred is not None:
        colors = [CENTERS_COLORS[i] for i in y_pred]
    else:
        colors = "black"
    ax.scatter(
        users_xy[:, 0],
        users_xy[:, 1],
        marker="x",
        label="patient",
        color=colors,
        s=30
    )

    # Draw links
    # if y_pred is not None:
    #     for i, j in enumerate(y_pred):
    #         ax.plot(
    #             [users_xy[i, 0], items_xy[j, 0]],
    #             [users_xy[i, 1], items_xy[j, 1]],
    #             color=CENTERS_COLORS[j],
    #             linewidth=.3,
    #             zorder=-1,
    #         )

    ax.grid(False)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_heatmap(y_true, y_pred, n_items, ax):
    m = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred
    )

    sns.heatmap(m,
                xticklabels=["Item {0}".format(i)
                             for i in range(1, n_items+1)],
                yticklabels=["Item {0}".format(i)
                             for i in range(1, n_items+1)],
                annot=True,
                cmap="YlGnBu",
                fmt=".0f",
                cbar=False,
                vmin=0,
                ax=ax)

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
