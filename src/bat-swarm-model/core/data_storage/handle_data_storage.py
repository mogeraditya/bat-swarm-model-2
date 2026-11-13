import gzip
import json
import pickle

import h5py
import numpy as np


def handle_data_storage_for_plotting(
    current_time,
    is_end_of_code,
    history,
    parameters_df,
    dir_to_store,
    store_data_for_plotting=True,
):
    """Generates files for data used for plotting.
    Periodically the history list is cleared to ensure
    RAM doesnt get used up.
    """
    if not store_data_for_plotting:
        return None
    else:
        history_array_size_limit = parameters_df["CLEANUP_PLOT_DATA"]

        if history and (len(history) >= history_array_size_limit or is_end_of_code):
            time_stamp = f"{current_time:.4f}".zfill(9)
            with gzip.open(
                dir_to_store + f"history_dump_{time_stamp}.pkl.gz",
                "wb",
                compresslevel=5,
            ) as f:
                pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

            # with open(dir_to_store + f"history_dump_{time_stamp}.pkl", "wb") as f:
            #     pickle.dump(history, f)

        if is_end_of_code:
            with open(
                dir_to_store + "/parameters_used.json", "w", encoding="utf-8"
            ) as fp:
                json.dump(parameters_df, fp)


# def handle_data_storage_for_scoring(
#     current_time,
#     is_end_of_code,
#     history,
#     parameters_df,
#     dir_to_store,
#     store_data_for_plotting=True,
# ):
#     """Generates files for data used for plotting.
#     Periodically the history list is cleared to ensure
#     RAM doesnt get used up.
#     """
#     history_array_size_limit = parameters_df["CLEANUP_PLOT_DATA"]

#     if len(history) >= history_array_size_limit or is_end_of_code:
#         # Save current batch as compressed numpy file instead of pickle
#         filename = dir_to_store + f"history_dump_{current_time:.3f}.npz"

#         # Convert history to compact numpy format
#         times = []
#         positions = []
#         directions = []
#         for frame in history:
#             times.append(frame["time"])
#             frame_positions = []
#             frame_directions = []
#             for bat_pos in frame["bat_positions"]:
#                 frame_positions.extend([bat_pos[0], bat_pos[1]])
#             for bat_dir in frame["bat_directions"]:
#                 frame_directions.extend([bat_dir[0], bat_dir[1]])
#             positions.append(frame_positions)
#             directions.append(frame_directions)

#         # Save as compressed numpy
#         times_array = np.array(times, dtype="f4")
#         max_bats = max(len(frame) // 2 for frame in positions) if positions else 0
#         positions_array = np.full((len(positions), max_bats * 2), np.nan, dtype="f4")
#         directions_array = np.full((len(directions), max_bats * 2), np.nan, dtype="f4")

#         for i, frame in enumerate(positions):
#             positions_array[i, : len(frame)] = frame
#         for i, frame in enumerate(directions):
#             directions_array[i, : len(frame)] = frame

#         np.savez_compressed(
#             filename,
#             times=times_array,
#             positions=positions_array,
#             directions=directions_array,
#         )

#     if is_end_of_code:
#         with open(dir_to_store + "/parameters_used.json", "w", encoding="utf-8") as fp:
#             json.dump(parameters_df, fp)


# def handle_data_storage_with_h5(
#     current_time,
#     is_end_of_code,
#     history,
#     parameters_df,
#     dir_to_store,
#     store_data_for_plotting=True,
# ):
#     history_array_size_limit = parameters_df["CLEANUP_PLOT_DATA"]

#     if len(history) >= history_array_size_limit or is_end_of_code:
#         # Save current batch as compressed numpy file
#         filename = dir_to_store + f"history_dump_{current_time:.3f}.npz"

#         # Collect all data for this batch
#         times = []
#         bat_positions = []  # List of [x, y] lists
#         bat_directions = []  # List of [dx, dy] lists
#         bat_response_vectors = []  # List of [vx, vy] lists
#         bat_ids = []
#         bat_ipi_counters = []
#         bat_call_times = []
#         bat_next_dir_angle = []
#         bat_response_types = []
#         group_pol_scores = []
#         group_local_pol_scores = []
#         bat_local_pol_scores = []
#         group_milling_scores = []

#         for frame in history:
#             t = frame["time"]
#             for bat_id in frame["bat_id"]:
#                 times.append(t)
#                 bat_positions.append(frame["bat_positions"][bat_id])
#                 bat_directions.append(frame["bat_directions"][bat_id])
#                 bat_response_vectors.append(frame["bat_response_vector"][bat_id])
#                 bat_ids.append(bat_id)
#                 bat_ipi_counters.append(frame["bat_ipi_counters"][bat_id])
#                 bat_call_times.append(frame["bat_call_time"][bat_id])
#                 bat_next_dir_angle.append(frame["next_dir_angle"][bat_id])
#                 bat_response_types.append(frame["response_type"][bat_id])
#                 group_pol_scores.append(frame["polarization_score"])
#                 group_local_pol_scores.append(frame["local_polarization_score"])
#                 bat_local_pol_scores.append(
#                     frame["local_polarization_score_list"][bat_id]
#                 )
#                 group_milling_scores.append(frame["milling_score"])

#         # Convert to numpy arrays
#         times_array = np.array(times, dtype="f4")
#         bat_positions_array = np.array(bat_positions, dtype="f4")
#         bat_directions_array = np.array(bat_directions, dtype="f4")
#         bat_response_vectors_array = np.array(bat_response_vectors, dtype="f4")
#         bat_ids_array = np.array(bat_ids, dtype="i4")
#         bat_ipi_counters_array = np.array(bat_ipi_counters, dtype="i4")
#         bat_call_times_array = np.array(bat_call_times, dtype="f4")
#         bat_next_dir_angle_array = np.array(bat_next_dir_angle, dtype="f4")
#         bat_response_types_array = np.array(bat_response_types, dtype="U20")
#         group_pol_scores_array = np.array(group_pol_scores, dtype="f4")
#         group_local_pol_scores_array = np.array(group_local_pol_scores, dtype="f4")
#         bat_local_pol_scores_array = np.array(bat_local_pol_scores, dtype="f4")
#         group_milling_scores_array = np.array(group_milling_scores, dtype="f4")

#         # Save all data in one compressed file
#         np.savez_compressed(
#             filename,
#             times=times_array,
#             bat_positions=bat_positions_array,
#             bat_directions=bat_directions_array,
#             bat_response_vectors=bat_response_vectors_array,
#             bat_ids=bat_ids_array,
#             bat_ipi_counters=bat_ipi_counters_array,
#             bat_call_times=bat_call_times_array,
#             bat_next_dir_angle=bat_next_dir_angle_array,
#             bat_response_types=bat_response_types_array,
#             group_pol_scores=group_pol_scores_array,
#             group_local_pol_scores=group_local_pol_scores_array,
#             bat_local_pol_scores=bat_local_pol_scores_array,
#             group_milling_scores=group_milling_scores_array,
#         )

#     if is_end_of_code:
#         with open(dir_to_store + "/parameters_used.json", "w", encoding="utf-8") as fp:
#             json.dump(parameters_df, fp)


def handle_data_storage(
    current_time,
    is_end_of_code,
    history,
    parameters_df,
    dir_to_store,
    store_data_for_plotting=True,
):

    # handle_data_storage_for_scoring(
    #     current_time,
    #     is_end_of_code,
    #     history,
    #     parameters_df,
    #     dir_to_store,
    #     store_data_for_plotting,
    # )
    handle_data_storage_for_plotting(
        current_time,
        is_end_of_code,
        history,
        parameters_df,
        dir_to_store,
        store_data_for_plotting,
    )
    # handle_data_storage_with_h5(
    #     current_time,
    #     is_end_of_code,
    #     history,
    #     parameters_df,
    #     dir_to_store,
    #     store_data_for_plotting,
    # )

    history_array_size_limit = parameters_df["CLEANUP_PLOT_DATA"]
    if len(history) >= history_array_size_limit or is_end_of_code:
        return True
    else:
        return False
