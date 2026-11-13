import numpy as np
import scipy as scp


def compute_nearest_neighbour_distance(bat_positions):
    """
    Calculate the mean nearest-neighbour distance across all bats and time frames.

    Args:
        bat_positions: Iterable of position arrays. Each array should have shape
            (n_bats, 2), where each row represents the x and y coordinates of a bat.

    Returns:
        float: The mean nearest-neighbour distance across every bat and time frame.
            Returns np.nan if no frame contains at least two bats.
    """
    nearest_neighbour_distances = []

    for position_frame in bat_positions:

        position_frame = np.asarray(position_frame)

        if len(position_frame) < 2:
            continue

        distance_matrix = scp.spatial.distance_matrix(position_frame, position_frame)

        np.fill_diagonal(distance_matrix, np.inf)

        frame_nearest_distances = np.min(distance_matrix, axis=1)

        nearest_neighbour_distances.append(frame_nearest_distances)

    if not nearest_neighbour_distances:
        return np.nan

    return nearest_neighbour_distances
