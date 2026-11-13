import glob
import pickle

import numpy as np
from tqdm.rich import tqdm


def stitch_together_history_lists(history_output_dir):
    """Merges lists from all the pickle files together.

    Args:
        history_output_dir (string):

    Returns:
        list: list with all the merged lists from pickle files.
    """
    history_output_dir = history_output_dir  # + "/data_for_plotting/"
    list_of_dict_files = glob.glob(history_output_dir + "/history_dump_*.pkl")
    list_of_dict_files = np.sort(list_of_dict_files)

    list_containing_data_from_all_pickle_files = []
    for pickle_file in list_of_dict_files:
        with open(pickle_file, "rb") as f:
            _list_containing_subset = pickle.load(f)
            list_containing_data_from_all_pickle_files.extend(_list_containing_subset)

    parameter_file = glob.glob(history_output_dir + "/parameters_used.json")[0]
    parameter_df = load_parameters(parameter_file)

    with open(history_output_dir + "/bats_initial.pkl", "rb") as f:
        bats_initial_positions = pickle.load(f)
    with open(history_output_dir + "/obstacles_initial.pkl", "rb") as f:
        obstacles_initial_positions = pickle.load(f)
    with open(history_output_dir + "/jammers_initial.pkl", "rb") as f:
        jammers_initial_positions = pickle.load(f)

    times = [i["time"] for i in list_containing_data_from_all_pickle_files]
    sorting_indices = np.argsort(times)
    list_containing_data_from_all_pickle_files = np.array(
        list_containing_data_from_all_pickle_files
    )
    list_containing_data_from_all_pickle_files = (
        list_containing_data_from_all_pickle_files[sorting_indices]
    )

    return (
        list_containing_data_from_all_pickle_files,
        parameter_df,
        bats_initial_positions,
        obstacles_initial_positions,
        jammers_initial_positions,
    )


def load_history_dump(folder_path):
    """Load and merge all history dump files in a folder by time order"""

    pattern = os.path.join(folder_path, "history_dump_*.npz")
    npz_files = glob.glob(pattern)

    if not npz_files:
        print(f"No history dump files found in {folder_path}")
        return []

    # extract timestamps from filenames and sort by time
    file_times = []
    for file_path in npz_files:
        try:

            filename = os.path.basename(file_path)
            time_str = filename.replace("history_dump_", "").replace(".npz", "")
            timestamp = float(time_str)
            file_times.append((timestamp, file_path))
        except ValueError:
            print(f"could not parse timestamp from filename: {filename}")
            continue

    file_times.sort(key=lambda x: x[0])
    all_frames = []

    for timestamp, file_path in tqdm(file_times, desc="Reading files"):
        # print(f"Loading: {os.path.basename(file_path)} (time: {timestamp})")

        data = np.load(file_path)
        times = data["times"]
        positions_array = data["positions"]
        directions_array = data["directions"]
        # call_times_array = data["bat_call_time"]
        for i, frame_time in enumerate(times):
            frame_data = positions_array[i]
            frame_direction_data = directions_array[i]

            valid_positions = frame_data[~np.isnan(frame_data)]
            valid_directions = frame_direction_data[~np.isnan(frame_direction_data)]

            bat_positions = np.array(
                [
                    (valid_positions[j], valid_positions[j + 1])
                    for j in range(0, len(valid_positions), 2)
                ]
            )
            bat_directions = np.array(
                [
                    (valid_directions[j], valid_directions[j + 1])
                    for j in range(0, len(valid_directions), 2)
                ]
            )
            # bat_call_time = np.array([])

            all_frames.append(
                {
                    "time": np.round(frame_time, 3),
                    "bat_positions": bat_positions[:],
                    "bat_directions": bat_directions[:],
                    # "bat_call_time": bat_call_time,
                }
            )
    all_frames.sort(key=lambda x: x["time"])
    return all_frames


def store_history_using_npz(history, dir_to_store, filename):
    # Save current batch as compressed numpy file instead of pickle
    filename = dir_to_store + f"{filename}.npz"

    # Convert history to compact numpy format
    times = []
    positions = []
    directions = []
    for frame in tqdm(history, desc=""):
        times.append(frame["time"])
        frame_positions = []
        frame_directions = []
        for bat_pos in frame["bat_positions"]:
            frame_positions.extend([bat_pos[0], bat_pos[1]])
        for bat_dir in frame["bat_directions"]:
            frame_directions.extend([bat_dir[0], bat_dir[1]])
        positions.append(frame_positions)
        directions.append(frame_directions)

    # Save as compressed numpy
    times_array = np.array(times, dtype="f4")
    max_bats = max(len(frame) // 2 for frame in positions) if positions else 0
    positions_array = np.full((len(positions), max_bats * 2), np.nan, dtype="f4")
    directions_array = np.full((len(directions), max_bats * 2), np.nan, dtype="f4")

    for i, frame in enumerate(positions):
        positions_array[i, : len(frame)] = frame
    for i, frame in enumerate(directions):
        directions_array[i, : len(frame)] = frame

    np.savez_compressed(
        filename,
        times=times_array,
        positions=positions_array,
        directions=directions_array,
    )
