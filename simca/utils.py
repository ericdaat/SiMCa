import torch
import numpy as np
import scipy


def distance_between_points(x1, y1, x2, y2, radius):
    # Straight distance
    straight_distance = np.sqrt(
        (y2 - y1)**2 + (x2 - x1)**2
    )

    # Great circle distance
    alpha = np.arctan2(y1, x1) - np.arctan2(y2, x2)
    circle_distance = radius * alpha
    circle_distance = abs(circle_distance)

    return straight_distance, circle_distance


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
