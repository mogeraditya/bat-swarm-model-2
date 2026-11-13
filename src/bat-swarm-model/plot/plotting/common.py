"""Shared helpers used by the individual plot modules.

These are lifted verbatim from the original monolithic plotting script -
only the *location* of the code changed, not the logic.
"""

import bisect
import glob
import gzip
import os
import pickle
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.rich import tqdm

sys.path.append("./src/bat-swarm-model/core/")

# sys.path.append("./scripts/")

from supporting_files.read_simulation_output import read_data_per_simulation_per_bat
from supporting_files.snr_implementation import parse_sounds, serialize_sound_info
from supporting_files.utilities import load_parameters

plt.style.use("dark_background")
# sys.path.append("./src")
plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"


def stitch_together_history_lists(history_output_dir):
    """Merges lists from all the pickle files together.

    Args:
        history_output_dir (string):

    Returns:
        list: list with all the merged lists from pickle files.
    """
    history_output_dir = history_output_dir  # + "/data_for_plotting/"
    list_of_dict_files = glob.glob(history_output_dir + "/history_dump_*.pkl.gz")
    list_of_dict_files = np.sort(list_of_dict_files)

    list_containing_data_from_all_pickle_files = []
    for pickle_file in tqdm(list_of_dict_files, desc="reading dict files "):
        with gzip.open(pickle_file, "rb") as f:
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


def plot_grid_matrix_into_radial_time_series(
    grid_matrix, spatial_grid_r, spatial_grid_theta, fig_and_ax, cmin, cmax
):
    r, theta = np.meshgrid(spatial_grid_r, spatial_grid_theta)
    z = grid_matrix.T.copy()
    fig, axs = fig_and_ax

    masked_z = np.ma.masked_where(z == 0, z)

    # Black to Blue to Cyan/Electric Blue
    colors = ["black", "red", "orange", "darkgreen", "green", "lime"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "black_to_cyan", colors, N=(cmax - cmin + 1)
    )
    cmap.set_bad(color="black", alpha=0)

    im = axs.pcolormesh(theta, r, masked_z, cmap=cmap, shading="auto")
    im.set_clim(cmin, cmax)
    fig.colorbar(im)
    axs.set_thetagrids(range(0, 360, int(360 / 12)))
    axs.set_theta_zero_location("N")

    return im


def increase_sensory_volume_resolution_for_plotting(
    sensory_volume_matrix, spatial_grid_theta, resolution
):
    sensory_volume_array = sensory_volume_matrix
    theta_space = spatial_grid_theta[1] - spatial_grid_theta[0]
    theta_for_increased_resolution = theta_space / resolution

    spatial_grid_theta_finer = np.arange(
        -np.pi, np.pi - theta_for_increased_resolution, theta_for_increased_resolution
    )
    sensory_volume_array_finer = np.zeros(shape=(len(spatial_grid_theta_finer)))

    for i, theta in enumerate(spatial_grid_theta_finer):
        index_in_old_data_array = i // resolution
        sensory_volume_array_finer[i] = sensory_volume_array[index_in_old_data_array]

    return sensory_volume_array_finer, spatial_grid_theta_finer


def plot_sensory_volume(sensory_volume_array, spatial_grid_theta, fig_and_ax):
    fig, axs = fig_and_ax

    line = axs.plot(
        spatial_grid_theta,
        sensory_volume_array,
        color="yellow",
        linewidth=2,
        label="Sound Intensity (dB)",
    )
    fill = axs.fill(spatial_grid_theta, sensory_volume_array, alpha=0.3, color="yellow")

    axs.set_thetagrids(range(0, 360, int(360 / 12)))
    axs.set_theta_zero_location("N")
    axs.set_rlim(0, 130)
    return line, fill
