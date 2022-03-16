import numpy as np
import seaborn as sns
from torch import nn


sns.set(style="whitegrid", palette="Set2")

ITEMS_COLORS = [
	"#D3F8E2",
	"#E4C1F9",
	"#F694C1",
	"#EDE7B1",
	"#A9DEF9",
	"#558564",
	"#564946",
	"#E53D00",
	"#046865",
	"#2C497F"
]


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

    # Plot users
    if y_pred is not None:
        colors = [ITEMS_COLORS[i] for i in y_pred]
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
        c=ITEMS_COLORS[:items_capacities.shape[1]],
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

    # Draw items
    ax.scatter(
        items_xy[:, 0],
        items_xy[:, 1],
        s=2*items_capacities,
        alpha=.7,
        label="care center",
        c=ITEMS_COLORS[:items_capacities.shape[1]],
        edgecolors="black"
    )

    if y_pred is not None:
        colors = [ITEMS_COLORS[i] for i in y_pred]
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

    ax.grid(False)
    ax.set_aspect("equal")
    ax.axis("off")
