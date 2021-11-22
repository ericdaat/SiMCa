import numpy as np


def compute_distance(x1, y1, x2, y2, radius):
    # Straight distance
    straight_distance = np.sqrt(
        (y2 - y1)**2 + (x2 - x1)**2
    )

    # Great circle distance
    alpha = np.arctan2(y1, x1) - np.arctan2(y2, x2)
    circle_distance = radius * alpha
    circle_distance = abs(circle_distance)

    return straight_distance, circle_distance
