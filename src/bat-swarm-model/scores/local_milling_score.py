# implementation from jitesh jawar + guttal et al 2019 noise induced schooling in fishes.
# breif exaplination
# take the unit direction vectors of all the agents and sum them up.
# score is the length of the resultant vector
import numpy as np
import scipy as scp


def compute_local_milling(
    bat_directions, bat_positions, parameters_df, neighbourhood_distance_threshold
):
    """compute the polarization of a frame of bat directions.
    Magnitude of the sum of the normalized direction vector of all the agents.

    Args:
        bat_directions (list): List containing Vectors of bat directions

    Returns:
        int: polarization score, ranges from 0 to 1
    """
    normalized_vectors = [i / np.linalg.norm(i) for i in bat_directions]
    pair_wise_distance_matrix = scp.spatial.distance_matrix(
        bat_positions, bat_positions
    )
    scores_for_neighbourhood_per_bat = []
    for i in range(pair_wise_distance_matrix.shape[0]):
        bats_within_neighbourhood = [i]
        directions_of_bats_within_neighbourhood = [normalized_vectors[i]]
        positions_of_bats_within_neighbourhood = [bat_positions[i]]
        for j in range(pair_wise_distance_matrix.shape[1]):
            if i != j:
                if pair_wise_distance_matrix[i, j] < neighbourhood_distance_threshold:
                    bats_within_neighbourhood.append(j)
                    directions_of_bats_within_neighbourhood.append(
                        normalized_vectors[j]
                    )
                    positions_of_bats_within_neighbourhood.append(bat_positions[j])
        # print(bats_within_neighbourhood)
        if len(bats_within_neighbourhood) <= 1:
            scores_for_neighbourhood_per_bat.append(0)
            break

        group_centriod = np.mean(positions_of_bats_within_neighbourhood, axis=0)

        difference_vector_from_mean = [
            i - group_centriod for i in positions_of_bats_within_neighbourhood
        ]
        normalized_difference_from_mean = [
            i / np.linalg.norm(i) for i in difference_vector_from_mean
        ]

        cross_product_with_direction = [
            np.cross(
                normalized_difference_from_mean[i],
                directions_of_bats_within_neighbourhood[i],
            )
            for i in range(len(normalized_difference_from_mean))
        ]

        mean_cross_vector = np.mean(cross_product_with_direction, axis=0)

        magnitude_of_mean_cross_vector = np.linalg.norm(mean_cross_vector)

        scores_for_neighbourhood_per_bat.append(magnitude_of_mean_cross_vector)

    local_polarization_score_for_group = np.mean(scores_for_neighbourhood_per_bat)

    return local_polarization_score_for_group


if __name__ == "__main__":
    x = [(0, 1), (1, 0)]
