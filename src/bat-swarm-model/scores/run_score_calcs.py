import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm.rich import tqdm

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)

import csv
import json
import math
import os

import numpy as np
from scipy.spatial.distance import cdist

# from post_processing_sweepers.condense_data import load_history_dump
from scores.calling_effort_score import calling_effort
from scores.collision_scores import compute_collision_counts_and_length
from scores.nearest_neighbour_distance import compute_nearest_neighbour_distance
from scores.space_occupied_scores import space_occupied_score
from scores.time_spent_rotating import compute_time_spent_rotating
from supporting_files.utilities import load_parameters

# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


# Change this value if a different sampling frequency is required.
DEFAULT_FRAME_INTERVAL = 1


# ===========================================================================
# Array helpers
# ===========================================================================
def filter_bat_positions_from_history(all_frames):

    store_only_positions = []
    store_only_directions = []

    for item in all_frames:
        # print(item)
        store_only_positions.append(item["bat_positions"])
        store_only_directions.append(item["bat_directions"])

    return store_only_positions, store_only_directions


def normalize_rows(vectors):
    """
    Normalize vectors row-by-row.

    Zero-length vectors remain zero instead of producing NaN values.
    """
    vectors = np.asarray(vectors, dtype=float)

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2D array of vectors, received shape {vectors.shape}"
        )

    if len(vectors) == 0:
        return vectors.copy()

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors, dtype=float),
        where=norms > 0,
    )


def validate_frame_data(bat_directions, bat_positions):
    """
    Convert frame data to arrays and validate their dimensions.
    """
    directions = np.asarray(bat_directions, dtype=float)
    positions = np.asarray(bat_positions, dtype=float)

    if directions.ndim != 2:
        raise ValueError(
            "bat_directions must be a 2D array. " f"Received shape {directions.shape}"
        )

    if positions.ndim != 2:
        raise ValueError(
            "bat_positions must be a 2D array. " f"Received shape {positions.shape}"
        )

    if len(directions) != len(positions):
        raise ValueError(
            "bat_directions and bat_positions must have the same "
            f"number of bats. Received {len(directions)} directions and "
            f"{len(positions)} positions."
        )

    if len(directions) > 0 and directions.shape[1] != positions.shape[1]:
        raise ValueError(
            "Directions and positions must have the same number of "
            f"dimensions. Received {directions.shape[1]} and "
            f"{positions.shape[1]}."
        )

    return directions, positions


def safe_nanmean(values):
    """
    Compute nanmean without warnings for an empty or all-NaN array.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmean(values))


def safe_nanstd(values):
    """
    Compute nanstd without warnings for an empty or all-NaN array.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan

    return float(np.nanstd(values))


def safe_nanmax(values):
    """
    Compute nanmax without warnings for an empty or all-NaN array.
    """
    values = np.asarray(values, dtype=float)

    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan

    return float(np.nanmax(values))


# ===========================================================================
# Global polarization
# ===========================================================================


def compute_polarization_from_normalized(normalized_directions):
    """
    Compute polarization from directions that are already normalized.
    """
    normalized_directions = np.asarray(
        normalized_directions,
        dtype=float,
    )

    if len(normalized_directions) == 0:
        return np.nan

    mean_direction = normalized_directions.mean(axis=0)

    return float(np.linalg.norm(mean_direction))


def compute_polarization(bat_directions):
    """
    Compute polarization for one frame.
    """
    directions = np.asarray(bat_directions, dtype=float)

    if directions.ndim != 2:
        raise ValueError(f"bat_directions must be 2D; received {directions.shape}")

    if len(directions) == 0:
        return np.nan

    normalized_directions = normalize_rows(directions)

    return compute_polarization_from_normalized(normalized_directions)


# ===========================================================================
# Milling
# ===========================================================================


def compute_milling_from_normalized(
    normalized_directions,
    bat_positions,
):
    """
    Compute milling using already-normalized direction vectors.
    """
    normalized_directions = np.asarray(
        normalized_directions,
        dtype=float,
    )
    positions = np.asarray(bat_positions, dtype=float)

    if len(positions) == 0:
        return np.nan

    centroid = positions.mean(axis=0)
    radial_vectors = positions - centroid
    normalized_radial_vectors = normalize_rows(radial_vectors)

    number_of_dimensions = positions.shape[1]

    if number_of_dimensions == 2:
        # Scalar z-component of a 2D cross product.
        cross_products = (
            normalized_radial_vectors[:, 0] * normalized_directions[:, 1]
            - normalized_radial_vectors[:, 1] * normalized_directions[:, 0]
        )

        return float(abs(cross_products.mean()))

    if number_of_dimensions == 3:
        cross_products = np.cross(
            normalized_radial_vectors,
            normalized_directions,
        )

        mean_cross_product = cross_products.mean(axis=0)

        return float(np.linalg.norm(mean_cross_product))

    raise ValueError(
        "Milling currently supports only 2D or 3D vectors. "
        f"Received {number_of_dimensions} dimensions."
    )


def compute_milling(bat_directions, bat_positions):
    """
    Compute milling for one frame.
    """
    directions, positions = validate_frame_data(
        bat_directions,
        bat_positions,
    )

    if len(positions) == 0:
        return np.nan

    normalized_directions = normalize_rows(directions)

    return compute_milling_from_normalized(
        normalized_directions,
        positions,
    )


# ===========================================================================
# Local polarization
# ===========================================================================


def compute_local_scores_from_distance_matrix(
    normalized_directions,
    squared_distance_matrix,
    neighbourhood_distance_threshold,
):
    """
    Compute per-bat local polarization and density using a precomputed
    squared-distance matrix.
    """
    threshold = float(neighbourhood_distance_threshold)

    if threshold <= 0:
        raise ValueError("Neighbourhood distance thresholds must be greater than zero.")

    normalized_directions = np.asarray(
        normalized_directions,
        dtype=float,
    )
    squared_distance_matrix = np.asarray(
        squared_distance_matrix,
        dtype=float,
    )

    number_of_bats = len(normalized_directions)

    if number_of_bats == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    if squared_distance_matrix.shape != (
        number_of_bats,
        number_of_bats,
    ):
        raise ValueError("The distance matrix shape does not match the number of bats.")

    # Preserve the original strict comparison:
    # distance < threshold.
    neighbourhood_mask = squared_distance_matrix < threshold * threshold

    # The diagonal is true, so each bat includes itself.
    neighbourhood_counts = neighbourhood_mask.sum(axis=1)

    # Sum the normalized directions in each neighborhood.
    summed_directions = neighbourhood_mask @ normalized_directions

    mean_directions = np.divide(
        summed_directions,
        neighbourhood_counts[:, np.newaxis],
        out=np.zeros_like(summed_directions, dtype=float),
        where=neighbourhood_counts[:, np.newaxis] > 0,
    )

    local_polarization = np.linalg.norm(
        mean_directions,
        axis=1,
    )

    # Preserve the behavior of the original implementation:
    # a bat with no other bats in its neighborhood has score zero.
    local_polarization[neighbourhood_counts <= 1] = 0.0

    neighbourhood_area = math.pi * threshold * threshold

    # Subtract one because a bat should not count itself toward density.
    local_density = (neighbourhood_counts - 1) / neighbourhood_area

    return local_polarization, local_density


def compute_local_polarization_per_bat(
    bat_directions,
    bat_positions,
    neighbourhood_distance_threshold,
):
    """
    Compatibility function for calculating local scores at one threshold.
    """
    directions, positions = validate_frame_data(
        bat_directions,
        bat_positions,
    )

    if len(positions) == 0:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    normalized_directions = normalize_rows(directions)

    squared_distance_matrix = cdist(
        positions,
        positions,
        metric="sqeuclidean",
    )

    return compute_local_scores_from_distance_matrix(
        normalized_directions,
        squared_distance_matrix,
        neighbourhood_distance_threshold,
    )


def aggregate_local_polarization(
    scores_for_neighbourhood_per_bat,
    local_density_list,
    type_id,
):
    """
    Aggregate per-bat local scores.
    """
    scores = np.asarray(
        scores_for_neighbourhood_per_bat,
        dtype=float,
    )
    densities = np.asarray(
        local_density_list,
        dtype=float,
    )

    if len(scores) != len(densities):
        raise ValueError(
            "Local polarization and density arrays must have equal lengths."
        )

    if len(scores) == 0:
        return np.nan, np.nan

    if type_id == "mean":
        return (
            float(scores.mean()),
            float(densities.mean()),
        )

    if type_id == "max":
        maximum_index = int(np.argmax(scores))

        return (
            float(scores[maximum_index]),
            float(densities[maximum_index]),
        )

    if (
        isinstance(type_id, (int, np.integer))
        and not isinstance(type_id, bool)
        and 0 <= type_id < len(scores)
    ):
        return (
            float(scores[type_id]),
            float(densities[type_id]),
        )

    raise TypeError("type_id must be 'mean', 'max', or a valid bat index.")


def compute_local_polarization(
    bat_directions,
    bat_positions,
    parameters_df,
    neighbourhood_distance_threshold,
    type_id,
):
    """
    Compatibility wrapper for the original function.

    parameters_df is retained in the signature but is not needed.
    """
    del parameters_df

    local_polarization, local_density = compute_local_polarization_per_bat(
        bat_directions,
        bat_positions,
        neighbourhood_distance_threshold,
    )

    return aggregate_local_polarization(
        local_polarization,
        local_density,
        type_id,
    )


# ===========================================================================
# Combined frame calculation
# ===========================================================================


def compute_frame_scores(
    bat_directions,
    bat_positions,
    neighbourhood_distances,
):
    """
    Compute all frame-based scores for one frame.

    Direction normalization and the pairwise distance matrix are each
    calculated only once.
    """
    directions, positions = validate_frame_data(
        bat_directions,
        bat_positions,
    )

    neighbourhood_distances = np.asarray(
        neighbourhood_distances,
        dtype=float,
    )

    if neighbourhood_distances.ndim != 1:
        raise ValueError("neighbourhood_distances must be one-dimensional.")

    if np.any(neighbourhood_distances <= 0):
        raise ValueError("Every neighbourhood distance must be greater than zero.")

    number_of_distances = len(neighbourhood_distances)

    if len(positions) == 0:
        return {
            "polarization": np.nan,
            "milling": np.nan,
            "local_polarization": np.full(
                number_of_distances,
                np.nan,
            ),
            "local_density": np.full(
                number_of_distances,
                np.nan,
            ),
        }

    # Normalize once and reuse.
    normalized_directions = normalize_rows(directions)

    polarization = compute_polarization_from_normalized(normalized_directions)

    milling = compute_milling_from_normalized(
        normalized_directions,
        positions,
    )

    local_polarization = np.empty(
        number_of_distances,
        dtype=float,
    )
    local_density = np.empty(
        number_of_distances,
        dtype=float,
    )

    if number_of_distances == 0:
        return {
            "polarization": polarization,
            "milling": milling,
            "local_polarization": local_polarization,
            "local_density": local_density,
        }

    # Compute the expensive distance matrix only once for this frame.
    squared_distance_matrix = cdist(
        positions,
        positions,
        metric="sqeuclidean",
    )

    for distance_index, distance in enumerate(neighbourhood_distances):
        per_bat_polarization, per_bat_density = (
            compute_local_scores_from_distance_matrix(
                normalized_directions,
                squared_distance_matrix,
                distance,
            )
        )

        local_polarization[distance_index] = per_bat_polarization.mean()
        local_density[distance_index] = per_bat_density.mean()

    return {
        "polarization": polarization,
        "milling": milling,
        "local_polarization": local_polarization,
        "local_density": local_density,
    }


# ===========================================================================
# Sampled frame calculations
# ===========================================================================


def create_sampled_frame_indices(
    number_of_frames,
    frame_interval=DEFAULT_FRAME_INTERVAL,
):
    """
    Return frame indices 0, interval, 2*interval, ...

    Example:
        number_of_frames = 350
        frame_interval = 100

        Result: [0, 100, 200, 300]
    """
    if frame_interval <= 0:
        raise ValueError("frame_interval must be greater than zero.")

    return np.arange(
        0,
        number_of_frames,
        frame_interval,
        dtype=int,
    )


def compute_sampled_frame_scores(
    bat_directions_list,
    bat_positions_list,
    neighbourhood_distances,
    frame_interval=DEFAULT_FRAME_INTERVAL,
):
    """
    Calculate frame-based scores only once every `frame_interval` frames.

    Returns:
        sampled_frame_indices
        polarization_array
        milling_array
        local_polarization_array
        local_density_array
    """
    if len(bat_directions_list) != len(bat_positions_list):
        raise ValueError("Direction and position histories must have equal lengths.")

    neighbourhood_distances = np.asarray(
        neighbourhood_distances,
        dtype=float,
    )

    if neighbourhood_distances.ndim != 1:
        raise ValueError("neighbourhood_distances must be one-dimensional.")

    number_of_frames = len(bat_positions_list)
    number_of_distances = len(neighbourhood_distances)

    sampled_frame_indices = create_sampled_frame_indices(
        number_of_frames,
        frame_interval,
    )

    number_of_sampled_frames = len(sampled_frame_indices)

    polarization_array = np.empty(
        number_of_sampled_frames,
        dtype=float,
    )
    milling_array = np.empty(
        number_of_sampled_frames,
        dtype=float,
    )
    local_polarization_array = np.empty(
        (number_of_sampled_frames, number_of_distances),
        dtype=float,
    )
    local_density_array = np.empty(
        (number_of_sampled_frames, number_of_distances),
        dtype=float,
    )

    for output_index, frame_index in enumerate(sampled_frame_indices):
        frame_scores = compute_frame_scores(
            bat_directions=bat_directions_list[frame_index],
            bat_positions=bat_positions_list[frame_index],
            neighbourhood_distances=neighbourhood_distances,
        )

        polarization_array[output_index] = frame_scores["polarization"]
        milling_array[output_index] = frame_scores["milling"]

        if number_of_distances > 0:
            local_polarization_array[output_index] = frame_scores["local_polarization"]
            local_density_array[output_index] = frame_scores["local_density"]

    return (
        sampled_frame_indices,
        polarization_array,
        milling_array,
        local_polarization_array,
        local_density_array,
    )


# ===========================================================================
# CSV helpers -- no pandas
# ===========================================================================


def convert_to_csv_value(value):
    """
    Convert NumPy and container values to CSV-compatible values.
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())

    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)

    return value


def write_timeseries_csv(
    output_path,
    sampled_frame_indices,
    timeseries_columns,
):
    """
    Write sampled time-series data directly with the csv module.

    Args:
        output_path:
            Destination CSV path.
        sampled_frame_indices:
            Original indices of the sampled frames.
        timeseries_columns:
            Dictionary mapping column names to sampled arrays.
    """
    sampled_frame_indices = np.asarray(
        sampled_frame_indices,
        dtype=int,
    )

    number_of_rows = len(sampled_frame_indices)

    for column_name, values in timeseries_columns.items():
        if len(values) != number_of_rows:
            raise ValueError(
                f"Column {column_name!r} has {len(values)} rows, "
                f"but {number_of_rows} rows were expected."
            )

    fieldnames = [
        "sample number",
        "frame index",
    ] + list(timeseries_columns.keys())

    with open(
        output_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for sample_number, frame_index in enumerate(sampled_frame_indices):
            row = {
                "sample number": sample_number,
                "frame index": int(frame_index),
            }

            for column_name, values in timeseries_columns.items():
                row[column_name] = convert_to_csv_value(values[sample_number])

            writer.writerow(row)


def write_summary_csv(output_path, summary_data):
    """
    Write a one-row summary CSV directly with the csv module.
    """
    fieldnames = list(summary_data.keys())

    row = {key: convert_to_csv_value(value) for key, value in summary_data.items()}

    with open(
        output_path,
        mode="w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerow(row)


def sample_existing_timeseries(
    values,
    sampled_frame_indices,
    column_name,
):
    """
    Sample an existing full-resolution time series at selected frame indices.
    """
    values = np.asarray(values)
    sampled_frame_indices = np.asarray(
        sampled_frame_indices,
        dtype=int,
    )

    if len(sampled_frame_indices) == 0:
        return np.empty(0, dtype=values.dtype)

    largest_index = int(sampled_frame_indices[-1])

    if len(values) <= largest_index:
        raise ValueError(
            f"Time series {column_name!r} has length {len(values)}, "
            f"but frame index {largest_index} was requested."
        )

    return values[sampled_frame_indices]


def parameters_to_dictionary(parameters):
    """
    Convert loaded parameters to a regular dictionary.

    `load_parameters` should preferably return a dictionary. Basic support
    for one-row DataFrame-like objects is retained without importing pandas.
    """
    if isinstance(parameters, dict):
        return parameters.copy()

    # Support a one-row pandas-like object if load_parameters still returns
    # one, without importing pandas here.
    if hasattr(parameters, "iloc") and hasattr(parameters, "__len__"):
        if len(parameters) != 1:
            raise ValueError("The parameter table must contain exactly one row.")

        return parameters.iloc[0].to_dict()

    if hasattr(parameters, "to_dict"):
        result = parameters.to_dict()

        if isinstance(result, dict):
            return result

    raise TypeError(
        "load_parameters must return a dictionary or a one-row "
        "DataFrame-like object."
    )


# ===========================================================================
# Main history function
# ===========================================================================


def given_history_list_generate_score_files(
    history_lists,
    data_dir,
    run_id,
    job_id,
    range_of_distance_for_local_score,
    frame_interval=DEFAULT_FRAME_INTERVAL,
):
    """
    Compute scores and save CSV files without pandas.

    Polarization, milling, local polarization, and local density are
    calculated only once every `frame_interval` frames.

    By default:
        frame_interval = 100

    Project-specific functions expected to already exist:

        filter_bat_positions_from_history
        load_parameters
        compute_collision_counts_and_length
        space_occupied_score
        compute_time_spent_rotating

    Returns:
        summary_data:
            A regular Python dictionary instead of a pandas DataFrame.
        batbat_collision_counter
        wall_collision_counter
    """
    print(
        f"Computing scores for {job_id} {run_id}; "
        f"one frame every {frame_interval} frames"
    )

    if history_lists is None or len(history_lists) == 0:
        raise ValueError("history_lists cannot be empty.")

    if frame_interval <= 0:
        raise ValueError("frame_interval must be greater than zero.")

    distance_covered = np.min(history_lists[-1]["bat_distance_covered"])

    (
        bat_positions_list,
        bat_directions_list,
    ) = filter_bat_positions_from_history(history_lists)

    # Preserve the behavior of the original code.
    # if len(bat_positions_list) > 10000:
    #     bat_positions_list = bat_positions_list[10000:]
    #     bat_directions_list = bat_directions_list[10000:]

    if len(bat_positions_list) != len(bat_directions_list):
        raise ValueError("Position and direction histories have different lengths.")

    duration = len(bat_positions_list)

    parameters_path = os.path.join(
        data_dir,
        "parameters_used.json",
    )
    parameters = load_parameters(parameters_path)

    # These calculations still use the complete position history so their
    # total counts retain their original meaning.
    (
        wall_collision_counter,
        batbat_collision_counter,
        timeseries_cumulative_wall_collision,
        timeseries_cumulative_bat_bat_collision,
    ) = compute_collision_counts_and_length(
        bat_positions_list,
        parameters,
    )

    space_occupied_counter = space_occupied_score(
        parameters,
        bat_positions_list,
        0,
    )

    neighbourhood_distances = np.asarray(
        range_of_distance_for_local_score,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Expensive frame scores are calculated only every 100 frames,
    # or according to the supplied frame_interval.
    # ---------------------------------------------------------------

    (
        sampled_frame_indices,
        polarization_array,
        milling_array,
        local_polarization_array,
        local_density_array,
    ) = compute_sampled_frame_scores(
        bat_directions_list=bat_directions_list,
        bat_positions_list=bat_positions_list,
        neighbourhood_distances=neighbourhood_distances,
        frame_interval=frame_interval,
    )

    print(
        f"Computed frame scores for {len(sampled_frame_indices)} "
        f"of {duration} frames"
    )

    # Sample the existing collision time series so every CSV column has
    # the same number of rows.
    sampled_batbat_collisions = sample_existing_timeseries(
        timeseries_cumulative_bat_bat_collision,
        sampled_frame_indices,
        "interbat collision cumulative",
    )

    sampled_wall_collisions = sample_existing_timeseries(
        timeseries_cumulative_wall_collision,
        sampled_frame_indices,
        "wall collision cumulative",
    )

    # ---------------------------------------------------------------
    # Construct sampled time-series columns
    # ---------------------------------------------------------------

    timeseries_columns = {
        "interbat collision cumulative": sampled_batbat_collisions,
        "wall collision cumulative": sampled_wall_collisions,
        "polarization": polarization_array,
        "milling": milling_array,
    }

    for distance_index, distance in enumerate(neighbourhood_distances):
        timeseries_columns[f"local polarization, range {distance:g}"] = (
            local_polarization_array[:, distance_index]
        )

        timeseries_columns[f"local density, range {distance:g}"] = local_density_array[
            :, distance_index
        ]

    # ---------------------------------------------------------------
    # Construct summary data
    # ---------------------------------------------------------------

    time_spent_rotating_score = compute_time_spent_rotating(
        history_lists,
        parameters,
    )

    summary_data = {
        "job id": job_id,
        "job uuid": run_id,
        "duration": duration,
        "frame score interval": frame_interval,
        "number of sampled frames": len(sampled_frame_indices),
        "distance_covered": distance_covered,
        "interbat collision total": batbat_collision_counter,
        "wall collision total": wall_collision_counter,
        "space_occupied": space_occupied_counter,
        "time_spent_rotating": safe_nanmax(time_spent_rotating_score),
        "mean polarization": safe_nanmean(polarization_array),
        "std polarization": safe_nanstd(polarization_array),
        "mean milling": safe_nanmean(milling_array),
        "std milling": safe_nanstd(milling_array),
    }

    for distance_index, distance in enumerate(neighbourhood_distances):
        distance_label = f"{distance:g}"

        polarization_timeseries = local_polarization_array[:, distance_index]
        density_timeseries = local_density_array[:, distance_index]

        summary_data[f"mean local polarization, range {distance_label}"] = safe_nanmean(
            polarization_timeseries
        )

        summary_data[f"std local polarization, range {distance_label}"] = safe_nanstd(
            polarization_timeseries
        )

        summary_data[f"mean local density, range {distance_label}"] = safe_nanmean(
            density_timeseries
        )

        summary_data[f"std local density, range {distance_label}"] = safe_nanstd(
            density_timeseries
        )

    # Add simulation parameters to the summary.
    parameter_dictionary = parameters_to_dictionary(parameters)

    for parameter_name, parameter_value in parameter_dictionary.items():
        summary_data[parameter_name] = parameter_value

    # ---------------------------------------------------------------
    # Write CSV files without pandas
    # ---------------------------------------------------------------

    os.makedirs(data_dir, exist_ok=True)

    timeseries_output_path = os.path.join(
        data_dir,
        f"timeseries_dataframe_job_{job_id}_run{run_id}.csv",
    )

    summary_output_path = os.path.join(
        data_dir,
        f"summary_dataframe_job_{job_id}_run{run_id}.csv",
    )

    print(f"Saving scores for {job_id} {run_id}")

    write_timeseries_csv(
        output_path=timeseries_output_path,
        sampled_frame_indices=sampled_frame_indices,
        timeseries_columns=timeseries_columns,
    )

    write_summary_csv(
        output_path=summary_output_path,
        summary_data=summary_data,
    )

    print(f"Saved scores for {job_id} {run_id}")

    return (
        summary_data,
        batbat_collision_counter,
        wall_collision_counter,
    )
