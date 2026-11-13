import numpy as np


def compute_milling(bat_directions, bat_positions):
    """Compute milling score of aa give frame of bat positions and directions
    Maagnitude of the sum of cross products between normalized direction vector
    of all the agents and the vector to the group centroid from the agents position.

    Args:
        bat_directions (list): list of Vectors of bat directions
        bat_positions (list): list of Vectors of bat positions

    Returns:
        int: milling score, ranges from 0-1
    """
    if len(bat_positions) == 0:
        return np.nan
    position_frame = bat_positions  # [i]
    normalized_vectors = [i / np.linalg.norm(i) for i in bat_directions]

    group_centriod = np.mean(position_frame, axis=0)

    difference_vector_from_mean = [i - group_centriod for i in position_frame]
    normalized_difference_from_mean = [
        i / np.linalg.norm(i) for i in difference_vector_from_mean
    ]

    cross_product_with_direction = [
        np.cross(normalized_difference_from_mean[i], normalized_vectors[i])
        for i in range(len(normalized_difference_from_mean))
    ]

    mean_cross_vector = np.mean(cross_product_with_direction, axis=0)

    magnitude_of_mean_cross_vector = np.linalg.norm(mean_cross_vector)
    return magnitude_of_mean_cross_vector
