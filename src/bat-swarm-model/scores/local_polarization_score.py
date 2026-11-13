# implementation from jitesh jawar + guttal et al 2019 noise induced schooling in fishes.
# breif exaplination
# take the unit direction vectors of all the agents and sum them up.
# score is the length of the resultant vector
import math

import numpy as np
import scipy as scp


def compute_local_polarization_per_bat(
    bat_directions, bat_positions, neighbourhood_distance_threshold
):
    """Local polarization and local density of every bat in one frame.

    Returns:
        (list, list): per-bat local polarization scores and local densities
    """
    normalized_vectors = [i / np.linalg.norm(i) for i in bat_directions]
    pair_wise_distance_matrix = scp.spatial.distance_matrix(
        bat_positions, bat_positions
    )
    scores_for_neighbourhood_per_bat = []
    neighbourhood_area = math.pi * (neighbourhood_distance_threshold**2)
    local_density_list = []
    for i in range(pair_wise_distance_matrix.shape[0]):
        bats_within_neighbourhood = [i]
        directions_of_bats_within_neighbourhood = [normalized_vectors[i]]
        for j in range(pair_wise_distance_matrix.shape[1]):
            if i != j:
                if pair_wise_distance_matrix[i, j] < neighbourhood_distance_threshold:
                    bats_within_neighbourhood.append(j)
                    directions_of_bats_within_neighbourhood.append(
                        normalized_vectors[j]
                    )
        # print(bats_within_neighbourhood)
        _local_density = (len(bats_within_neighbourhood) - 1) / neighbourhood_area
        local_density_list.append(_local_density)

        if len(bats_within_neighbourhood) <= 1:
            scores_for_neighbourhood_per_bat.append(0)
            continue

        summed_vector_for_neighbourhood = np.mean(
            directions_of_bats_within_neighbourhood, axis=0
        )
        magnitude_of_summed_vector_for_neighbourhood = np.linalg.norm(
            summed_vector_for_neighbourhood
        )
        scores_for_neighbourhood_per_bat.append(
            magnitude_of_summed_vector_for_neighbourhood
        )
    return scores_for_neighbourhood_per_bat, local_density_list


def compute_local_polarization(
    bat_directions,
    bat_positions,
    parameters_df,
    neighbourhood_distance_threshold,
    type_id,
):
    """compute the polarization of a frame of bat directions.
    Magnitude of the sum of the normalized direction vector of all the agents.

    Args:
        bat_directions (list): List containing Vectors of bat directions

    Returns:
        int: polarization score, ranges from 0 to 1
    """
    scores_for_neighbourhood_per_bat, local_density_list = (
        compute_local_polarization_per_bat(
            bat_directions, bat_positions, neighbourhood_distance_threshold
        )
    )
    return aggregate_local_polarization(
        scores_for_neighbourhood_per_bat, local_density_list, type_id
    )


def aggregate_local_polarization(
    scores_for_neighbourhood_per_bat, local_density_list, type_id
):
    """reduce per-bat scores to the group mean / max / one bat's value"""
    if type_id == "mean":
        local_polarization_score_for_group = np.mean(scores_for_neighbourhood_per_bat)
        local_density = np.mean(local_density_list)
    elif type_id == "max":
        local_polarization_score_for_group = np.max(scores_for_neighbourhood_per_bat)
        local_density = local_density_list[np.argmax(scores_for_neighbourhood_per_bat)]
    elif (
        isinstance(type_id, int)
        and type_id >= 0
        and type_id <= len(scores_for_neighbourhood_per_bat)
    ):
        local_polarization_score_for_group = scores_for_neighbourhood_per_bat[type_id]
        local_density = local_density_list[type_id]
    else:
        raise TypeError("Input is not valid (not mean or the bat id!)")

    return local_polarization_score_for_group, local_density


if __name__ == "__main__":
    x = [(0, 1), (1, 0)]
