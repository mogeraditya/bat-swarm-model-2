import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm.rich import tqdm

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)

# from post_processing_sweepers.condense_data import load_history_dump
from scores.calling_effort_score import calling_effort
from scores.collision_scores import compute_collision_counts_and_length
from scores.local_polarization_score import compute_local_polarization
from scores.milling_score import compute_milling
from scores.nearest_neighbour_distance import compute_nearest_neighbour_distance
from scores.polarization_score import compute_polarization
from scores.space_occupied_scores import space_occupied_score
from scores.time_spent_rotating import compute_time_spent_rotating
from supporting_files.utilities import load_parameters


def filter_bat_positions_from_history(all_frames):

    store_only_positions = []
    store_only_directions = []

    for item in all_frames:
        # print(item)
        store_only_positions.append(item["bat_positions"])
        store_only_directions.append(item["bat_directions"])

    return store_only_positions, store_only_directions


def reformat_history(history, focal_bat, parameter_df, iteration_label):
    # get time, position, call time data
    # reformat into long csv
    subset_of_focal_bat = []
    for item in history:
        reformatted_item = {
            "time": item["time"],
            "bat_position_x": item["bat_positions"][focal_bat][0],
            "bat_position_y": item["bat_positions"][focal_bat][1],
            "bat_call_time": item["bat_call_time"][focal_bat],
            "iteration_number": iteration_label,
        }
        reformatted_item.update(parameter_df.copy())
        subset_of_focal_bat.append(reformatted_item)

    # df_position_data = pd.DataFrame.from_dict(subset_of_focal_bat, orient="columns")
    # print(df_position_data)
    return subset_of_focal_bat


def take_history_store_scores(sim, sim_iteration_number):
    focal_bat = sim.bats[0]
    bat_positions = filter_bat_positions_from_history(sim.history)

    space_occupied = space_occupied_score(sim.parameters_df, bat_positions, 0)
    collision_counts = compute_collision_counts_and_length(
        bat_positions, sim.parameters_df
    )
    collision_rate = compute_collision_counts_and_length(
        bat_positions, sim.parameters_df
    )
    calling_effort_score = calling_effort(focal_bat)

    dict_to_store = {
        "space_occupied": space_occupied,
        "collision_counts": collision_counts,
        "collision_rate": collision_rate,
        "calling_effort": calling_effort_score,
        "iteration_label": sim_iteration_number,
    }
    dict_to_store.update(sim.parameters_df)

    return dict_to_store


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def given_history_list_generate_score_files(
    history_lists,
    data_dir,
    run_id,
    job_id,
    range_of_distance_for_local_score,
):
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

    list_of_score_summary = [
        [job_id],
        [run_id],
        [duration],
        [distance_covered],
        [batbat_collision_counter],
        [wall_collision_counter],
        [space_occupied_counter],
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

    dataframe_timeseries.to_csv(
        data_dir + f"/timeseries_dataframe_job_{job_id}_run{run_id}.csv"
    )
    dataframe_score_summary.to_csv(
        data_dir + f"/summary_dataframe_job_{job_id}_run{run_id}.csv"
    )

    return dataframe_score_summary, batbat_collision_counter, wall_collision_counter


def given_history_list_generate_score_files_for_sensitivity(
    history_lists,
    data_dir,
    run_id,
    job_id,
    range_of_distance_for_local_score,
):
    distance_covered = history_lists[-1]["bat_distance_covered"][0]
    bat_positions_list, bat_directions_list = filter_bat_positions_from_history(
        history_lists
    )
    parameters_df = load_parameters(data_dir + "/parameters_used.json")
    duration = len(bat_positions_list)
    (
        wall_collision_counter,
        batbat_collision_counter,
        timeseries_cumulative_wall_collision,
        timeseries_cumulative_bat_bat_collision,
    ) = compute_collision_counts_and_length(bat_positions_list, parameters_df)
    space_occupied_counter = space_occupied_score(parameters_df, bat_positions_list, 0)
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
    ]

    list_of_summary_keys = [
        "job id",
        "job uuid",
        "duration",
        "distance travelled",
        "interbat collision total",
        "wall collision total",
        "space_occupied",
        "time_spent_rotating",
    ]

    dict_score_summary = dict(zip(list_of_summary_keys, list_of_score_summary))

    merged_dict_with_params = dict_score_summary | parameters_df
    dataframe_score_summary = pd.DataFrame.from_dict(merged_dict_with_params)

    dataframe_score_summary.to_csv(data_dir + f"/summary_dataframe_job_{job_id}.csv")
    return (dataframe_score_summary, batbat_collision_counter, wall_collision_counter)


# if __name__ == "__main__":

#     output_dir = r"/home/adityamoger/Documents/GitHub/bat-swarm-model/multirun/2026-09-02/17-52-52/15/data/"
#     bat_positions_raw = load_history_dump(output_dir)
#     bat_positions, bat_directions = filter_bat_positions_from_history(bat_positions_raw)
#     x = np.array(bat_directions, dtype=np.half)
#     print(x.shape)

#     import matplotlib.pyplot as plt

#     def running_mean(x, N):
#         cumsum = np.cumsum(np.insert(x, 0, 0))
#         return (cumsum[N:] - cumsum[:-N]) / float(N)

#     milling_score_timeseries = compute_milling(
#         bat_directions=bat_directions, bat_positions=bat_positions
#     )
#     polarization_score_timeseries = compute_polarization(bat_directions)
#     nnd_timeseries = compute_nearest_neighbour_distance(bat_positions)
#     median_nnd_time_series = np.median(nnd_timeseries, axis=1)

#     milling_score_timeseries = running_mean(milling_score_timeseries, 1000)
#     polarization_score_timeseries = running_mean(polarization_score_timeseries, 1000)
#     median_nnd_time_series = running_mean(median_nnd_time_series, 1000)
#     plt.subplot(1, 2, 1)
#     plt.plot(milling_score_timeseries, label="milling")
#     plt.plot(polarization_score_timeseries, label="polarization")
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(median_nnd_time_series, label="nnd")
#     plt.legend()
#     plt.show()
