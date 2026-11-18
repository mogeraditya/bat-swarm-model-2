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


# implementation from jitesh jawar + guttal et al 2019 noise induced schooling in fishes.
# breif exaplination
# take the unit direction vectors of all the agents and sum them up.
# score is the length of the resultant vector
import numpy as np


def compute_polarization(bat_directions):
    """compute the polarization of a frame of bat directions.
    Magnitude of the sum of the normalized direction vector of all the agents.

    Args:
        bat_directions (list): List containing Vectors of bat directions

    Returns:
        int: polarization score, ranges from 0 to 1
    """
    normalized_vectors = [i / np.linalg.norm(i) for i in bat_directions]
    summed_vector = np.mean(normalized_vectors, axis=0)
    magnitude_of_summed_vector = np.linalg.norm(summed_vector)
    return magnitude_of_summed_vector


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


def given_history_list_generate_score_files(
    history_lists,
    data_dir,
    run_id,
    job_id,
    range_of_distance_for_local_score,
):
    print(f"computing scores for {job_id} {run_id}")
    distance_covered = history_lists[-1]["bat_distance_covered"][0]
    bat_positions_list, bat_directions_list = filter_bat_positions_from_history(
        history_lists
    )
    if len(bat_positions_list) > 10000:
        bat_positions_list = bat_positions_list[10000:]
        bat_directions_list = bat_directions_list[10000:]
    parameters_df = load_parameters(data_dir + "/parameters_used.json")
    duration = len(bat_positions_list)
    (
        wall_collision_counter,
        batbat_collision_counter,
        timeseries_cumulative_wall_collision,
        timeseries_cumulative_bat_bat_collision,
    ) = compute_collision_counts_and_length(bat_positions_list, parameters_df)
    space_occupied_counter = space_occupied_score(parameters_df, bat_positions_list, 0)
    polarization_list = [
        compute_polarization(bat_directions) for bat_directions in bat_directions_list
    ]
    milling_list = [
        compute_milling(bat_directions, bat_positions_list[i])
        for i, bat_directions in enumerate(bat_directions_list)
    ]

    list_of_local_polarization_timeseries = []
    list_of_local_polarization_summary = []
    keys_for_local_polarization = []
    keys_for_local_polarization_summary = []
    for distance in range_of_distance_for_local_score:
        local_polarization_list = [
            compute_local_polarization(
                bat_directions, bat_positions_list[i], parameters_df, distance, "mean"
            )
            for i, bat_directions in enumerate(bat_directions_list)
        ]
        list_of_local_polarization_timeseries.append(local_polarization_list)
        keys_for_local_polarization.append(f"local polarization, range {distance}")

        list_of_local_polarization_summary.extend(
            [[np.mean(local_polarization_list)], [np.std(local_polarization_list)]]
        )
        keys_for_local_polarization_summary.extend(
            [
                f"mean local polarization, range {distance}",
                f"std local polarization, range {distance}",
            ]
        )

    list_of_timeseries_scores = [
        timeseries_cumulative_bat_bat_collision,
        timeseries_cumulative_wall_collision,
        polarization_list,
        milling_list,
    ]
    list_of_keys_timeseries = [
        "interbat collision cumulative",
        "wall collision cumulative",
        "polarization",
        "milling",
    ]

    list_of_timeseries_scores.extend(list_of_local_polarization_timeseries)
    list_of_keys_timeseries.extend(keys_for_local_polarization)

    dataframe_timeseries = dict(zip(list_of_keys_timeseries, list_of_timeseries_scores))
    dataframe_timeseries = pd.DataFrame.from_dict(dataframe_timeseries)

    time_spent_rotating_score = compute_time_spent_rotating(
        history_lists, parameters_df
    )

    list_of_score_summary = [
        [job_id],
        [run_id],
        [duration],
        [distance_covered],
        [batbat_collision_counter],
        [wall_collision_counter],
        [space_occupied_counter],
        [np.max(time_spent_rotating_score)],
        [np.mean(polarization_list)],
        [np.std(polarization_list)],
        [np.mean(milling_list)],
        [np.std(milling_list)],
    ]

    list_of_summary_keys = [
        "job id",
        "job uuid",
        "duration",
        "distance_covered",
        "interbat collision total",
        "wall collision total",
        "space_occupied",
        "time_spent_rotating",
        "mean polarization",
        "std polarization",
        "mean milling",
        "std milling",
    ]

    list_of_score_summary.extend(list_of_local_polarization_summary)
    list_of_summary_keys.extend(keys_for_local_polarization_summary)

    dict_score_summary = dict(zip(list_of_summary_keys, list_of_score_summary))

    merged_dict_with_params = dict_score_summary | parameters_df
    dataframe_score_summary = pd.DataFrame.from_dict(merged_dict_with_params)

    print(f"done computing, saving scores for {job_id} {run_id}")
    dataframe_timeseries.to_csv(
        data_dir + f"/timeseries_dataframe_job_{job_id}_run{run_id}.csv"
    )
    dataframe_score_summary.to_csv(
        data_dir + f"/summary_dataframe_job_{job_id}_run{run_id}.csv"
    )
    print(f"Saved! scores for {job_id} {run_id}")
    return dataframe_score_summary, batbat_collision_counter, wall_collision_counter
