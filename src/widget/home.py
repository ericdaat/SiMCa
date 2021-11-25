import base64
from io import BytesIO
import logging

import torch
import numpy as np
from matplotlib.figure import Figure
import seaborn as sns
from flask import Blueprint, render_template, request

from src.dataset.toy import ToyDataset, CIRCLE_RADIUS
from src import viz
from src.ml import train


bp = Blueprint("home", __name__)

toy = ToyDataset(
    n_users=1000,
    n_pois=3,
    n_centers=3,
    n_features=2,
    distance_weight=0
)

@bp.route("/")
def index():
    # dataset parameters
    n_users = request.args.get("n_users", default=1000, type=int)
    n_pois = request.args.get("n_pois", default=3, type=int)
    n_centers = request.args.get("n_centers", default=3, type=int)
    distance_weight = request.args.get("distance_weight", default=0, type=float)
    generate = request.args.get("generate", default=False, type=bool)

    # model parameters
    user_embeddings = request.args.get("users_embeddings", default="pre-trained", type=str)
    assign = request.args.get("assign", default="lap", type=str)
    n_epochs = request.args.get("n_epochs", default=200, type=int)
    epsilon = request.args.get("epsilon", default=0.1, type=float)
    n_iter = request.args.get("n_iter", default=10, type=int)
    alpha = request.args.get("alpha", default=0, type=float)
    lr = request.args.get("lr", default=0.01, type=float)

    ####################
    # Generate dataset #
    ####################
    # if generation is asked or if dataset parameters changed
    if (generate or n_users!=toy.n_users or n_pois!=toy.n_pois
        or n_centers!=toy.n_centers or distance_weight!=toy.distance_weight):
        logging.info("Drawing a new dataset")
        toy.update_dataset(
            n_users=n_users,
            n_pois=n_pois,
            n_centers=n_centers,
            n_features=2,
            distance_weight=distance_weight
        )

    ###############
    # Train model #
    ###############

    if user_embeddings == "learn":
        users_features = None
        train_user_embeddings = True
    else:
        users_features = toy.users_features
        train_user_embeddings = False

    (y_pred, model, losses_df, scores_df, capacities_df) \
        = train.train_model(
            users_tensor=toy.users_tensor,
            pois_tensor=toy.pois_tensor,
            D_tensor=toy.D_tensor,
            y_true_tensor=toy.y_true_tensor,
            pois_capacities=toy.pois_capacities,
            items_features=toy.pois_features,
            n_features=2,
            lr=lr,
            epsilon=epsilon,
            n_iter=n_iter,
            alpha=alpha,
            n_epochs=n_epochs,
            assign=assign,
            users_features=users_features,
            train_user_embeddings=train_user_embeddings
        )

    #######################
    # Plot results: Fig 1 #
    #######################
    fig1 = Figure(figsize=(10, 5))
    axs = fig1.subplots(nrows=1, ncols=2)

    # Ground truth
    viz.plot_embeddings(
        users_features=torch.from_numpy(toy.users_features),
        pois_features=torch.from_numpy(toy.pois_features),
        pois_capacities=toy.pois_capacities,
        y_pred=toy.assigned_poi_for_user,
        ax=axs[0]
    )

    viz.plot_distances(
        np.asarray(list(zip(toy.users_x, toy.users_y))),
        np.asarray(list(zip(toy.pois_x, toy.pois_y))),
        CIRCLE_RADIUS,
        toy.assigned_poi_for_user,
        toy.pois_capacities,
        axs[1]
    )

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig1.savefig(buf, format="png")

    # Embed the result in the html output.
    fig1_data = base64.b64encode(buf.getbuffer()).decode("ascii")

    #######################
    # Plot results: Fig 2 #
    #######################
    fig2 = Figure(figsize=(10, 5))
    axs = fig2.subplots(nrows=1, ncols=2)

    # Predictions
    viz.plot_embeddings(
        users_features=model.user_embeddings.weight,
        pois_features=model.poi_embeddings.weight,
        pois_capacities=toy.pois_capacities,
        y_pred=y_pred,
        ax=axs[0]
    )

    viz.plot_distances(
        np.asarray(list(zip(toy.users_x, toy.users_y))),
        np.asarray(list(zip(toy.pois_x, toy.pois_y))),
        CIRCLE_RADIUS,
        y_pred,
        toy.pois_capacities,
        axs[1]
    )

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig2.savefig(buf, format="png")

    # Embed the result in the html output.
    fig2_data = base64.b64encode(buf.getbuffer()).decode("ascii")

    #######################
    # Plot results: Fig 3 #
    #######################
    fig3 = Figure(figsize=(9, 10))
    axs = fig3.subplots(nrows=2, ncols=2)

    viz.plot_scores(scores_df, axs[0][0])
    viz.plot_losses(losses_df, axs[0][1])
    viz.plot_capacities(capacities_df, axs[1][0])
    viz.plot_heatmap(toy.y_true_tensor, y_pred, n_pois, axs[1][1])

    fig3.tight_layout()

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig3.savefig(buf, format="png")

    # Embed the result in the html output.
    fig3_data = base64.b64encode(buf.getbuffer()).decode("ascii")

    return render_template(
        "home.html",
        fig1_data=fig1_data,
        fig2_data=fig2_data,
        fig3_data=fig3_data,
        pois_colors=viz.CENTERS_COLORS[:n_pois]
    )
