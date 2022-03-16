import torch
import numpy as np
from sklearn.metrics import f1_score

from simca.utils import assign_with_lap


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

            P = sinkhorn_loss.solver(
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

        if epoch_number == 1 or epoch_number % 20 == 0:
            print("Epoch {0}: loss={1:.3f}, f1={2:.3f}, avg_dist={3:.3f}"
                  .format(epoch_number, loss.item(), f1, average_distance))

        training_results.append([
            epoch_number,
            loss.item(),
            f1,
            average_distance
        ])

    return training_results
