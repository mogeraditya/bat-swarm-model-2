import bisect
import sys

import numpy as np

sys.path.append("./src/")


def find_sum_of_db(list_of_spls):
    """Computes the sum of all the spls in a given list.

    Args:
        list_of_spls (list): contains all the spls (in dB scale) that need to be added

    Returns:
        float: sum of spls in the list (in dB scale)
    """
    _temporary_sum = sum(10 ** (spl / 20) for spl in list_of_spls if spl != 0)
    return 0 if _temporary_sum == 0 else np.round(20 * np.log10(_temporary_sum), 2)


def generate_empty_matrix(parameters_df):
    """generate an empty matrix to store sensory volume information

    Args:
        parameters_df (dict): parameter file used for the simulations

    Returns:
        tuple : first matrix is the empty matrix and
        second is the reference list of angles for columns.
    """
    angular_resolution = np.radians(parameters_df["BAT_ANGULAR_RESOLUTION"])
    spatial_grid_theta = np.arange(-np.pi, np.pi, angular_resolution)
    return np.zeros((1, len(spatial_grid_theta))), spatial_grid_theta


def convert_angles_to_negpi_to_pi(list_of_angles):
    return np.array(
        [
            angle % (-np.sign(angle) * np.pi) if abs(angle) > np.pi else angle
            for angle in list_of_angles
        ]
    )


def find_indices_corresponding_to_hearing_range(
    parameters_df,
    spatial_grid_theta,
    angle_between_reference_axis_and_bat,
    hearing_range,
):
    bat_angular_resolution = np.radians(parameters_df["BAT_ANGULAR_RESOLUTION"])
    angles_to_hearing_range = np.arange(
        -hearing_range + angle_between_reference_axis_and_bat,
        hearing_range
        + angle_between_reference_axis_and_bat
        + bat_angular_resolution / 2,
        bat_angular_resolution,
    )
    angles_to_hearing_range = convert_angles_to_negpi_to_pi(angles_to_hearing_range)
    return [
        bisect.bisect_left(spatial_grid_theta, angle) % len(spatial_grid_theta)
        for angle in angles_to_hearing_range
    ]


def modify_matrix_based_on_hearing_range(
    matrix,
    angle_between_reference_axis_and_bat,
    parameters_df,
    angle_threshold,
):
    angular_resolution = np.radians(parameters_df["BAT_ANGULAR_RESOLUTION"])
    spatial_grid_theta = np.arange(-np.pi, np.pi, angular_resolution)
    hearing_range = np.radians(angle_threshold)
    hearing_range_indices = find_indices_corresponding_to_hearing_range(
        parameters_df,
        spatial_grid_theta,
        angle_between_reference_axis_and_bat,
        hearing_range,
    )
    matrix_based_on_hearing_range = np.zeros_like(matrix)
    matrix_based_on_hearing_range[0, hearing_range_indices] = matrix[
        0, hearing_range_indices
    ]
    return matrix_based_on_hearing_range, np.array(hearing_range_indices)


def convert_detected_sounds_into_grids_loudest_sound(
    heard_sounds, parameters_df, bat_direction, allocentric_axis_y
):
    spatial_reference_frame = parameters_df["SPATIAL_REFERENCE_FRAME"]
    matrix_spatial_grid, spatial_grid_theta = generate_empty_matrix(parameters_df)
    store_grid = matrix_spatial_grid.copy()
    store_grid_sum = matrix_spatial_grid.copy()

    if len(heard_sounds) == 0:
        return store_grid_sum, spatial_grid_theta

    for sound_object in heard_sounds:
        if spatial_reference_frame == "allocentric":
            theta = allocentric_axis_y.angle_between(sound_object["incident_direction"])
        elif spatial_reference_frame == "egocentric":
            theta = bat_direction.angle_between(sound_object["incident_direction"])
        else:
            raise ValueError("not a supported spatial reference frame")

        grid_column_index = bisect.bisect_left(spatial_grid_theta, theta) % len(
            spatial_grid_theta
        )
        sound_max_intensity = np.max(sound_object["all_spl_values"])

        store_grid[0, grid_column_index] = max(
            store_grid[0, grid_column_index], sound_max_intensity
        )
        store_grid_sum[0, grid_column_index] = find_sum_of_db(
            [store_grid_sum[0, grid_column_index], sound_max_intensity]
        )

    return store_grid, spatial_grid_theta


def select_cell_to_respond_to(matrix, intensity_threshold, neighbour_selection):
    """pick the cell to respond to among the ones above the intensity threshold

    Args:
        matrix (np.ndarray): 1 x n sensory grid restricted to the hearing range
        intensity_threshold (float): spl a cell must exceed to be a candidate
        neighbour_selection (str): "loudest" picks the loudest cell,
            "random" picks uniformly among the cells above threshold

    Returns:
        int: column index of the selected cell
    """
    if neighbour_selection == "loudest":
        return np.argmax(matrix)
    if neighbour_selection == "random":
        return np.random.choice(np.flatnonzero(matrix > intensity_threshold))
    raise ValueError(
        "NEIGHBOUR_SELECTION should be either 'loudest' or 'random' in the parameters file."
    )


def given_matrix_find_cell_to_respond_to_loudest_sound(
    matrix,
    parameters_df,
    bat_direction,
    allocentric_axis_y,
    id,
    previous_output_cell,
):
    matrix = np.matrix(matrix)
    angle_between_reference_axis_and_bat = allocentric_axis_y.angle_between(
        bat_direction
    )

    matrix_based_on_attraction_range, attraction_range_indices = (
        modify_matrix_based_on_hearing_range(
            matrix,
            angle_between_reference_axis_and_bat,
            parameters_df,
            parameters_df["ANGLE_RANGE_FOR_ATTRACTION"],
        )
    )
    matrix_based_on_repulsion_range, repulsion_range_indices = (
        modify_matrix_based_on_hearing_range(
            matrix,
            angle_between_reference_axis_and_bat,
            parameters_df,
            parameters_df["ANGLE_RANGE_FOR_REPULSION"],
        )
    )

    intensity_threshold_for_attraction = parameters_df["SPL_THRESHOLD_FOR_ATTRACTION"]
    intensity_threshold_for_repulsion = parameters_df["SPL_THRESHOLD_FOR_REPULSION"]
    neighbour_selection = parameters_df.get("NEIGHBOUR_SELECTION", "random")

    is_bat_doing_repulsion = np.any(
        matrix_based_on_repulsion_range > intensity_threshold_for_repulsion
    )
    is_bat_doing_attraction = np.any(
        matrix_based_on_attraction_range > intensity_threshold_for_attraction
    )

    if is_bat_doing_repulsion:
        if (
            previous_output_cell[1] in repulsion_range_indices
            and matrix_based_on_repulsion_range[0, previous_output_cell[1]]
            > intensity_threshold_for_repulsion
        ):
            return 0, previous_output_cell[1]

        selected_cell_in_repulsion_matrix = select_cell_to_respond_to(
            matrix_based_on_repulsion_range,
            intensity_threshold_for_repulsion,
            neighbour_selection,
        )
        output_cell_number = 0, selected_cell_in_repulsion_matrix

    elif is_bat_doing_attraction:
        if (
            previous_output_cell[1] in attraction_range_indices
            and matrix_based_on_attraction_range[0, previous_output_cell[1]]
            > intensity_threshold_for_attraction
        ):
            return 0, previous_output_cell[1]

        selected_cell_in_attraction_matrix = select_cell_to_respond_to(
            matrix_based_on_repulsion_range,
            intensity_threshold_for_attraction,
            neighbour_selection,
        )
        output_cell_number = 0, selected_cell_in_attraction_matrix

    elif not (is_bat_doing_repulsion or is_bat_doing_attraction):
        output_cell_number = np.nan, np.nan

    else:
        print(is_bat_doing_repulsion, is_bat_doing_attraction)
        raise ValueError(
            "the agent is attracting and repelling at the same time! impossible!!!"
        )

    return output_cell_number


def given_time_and_angle_return_direction_loudest_sound(
    intensity,
    angle_of_activated_cell,
    parameters_df,
    bat_direction,
    allocentric_axis_y,
):
    spatial_reference_frame = parameters_df["SPATIAL_REFERENCE_FRAME"]
    angular_resolution = np.radians(parameters_df["BAT_ANGULAR_RESOLUTION"])
    noise_in_angle = np.random.uniform(-angular_resolution / 2, angular_resolution / 2)
    angle_of_next_direction = (
        angle_of_activated_cell - angular_resolution / 2 + noise_in_angle
    )

    if spatial_reference_frame == "allocentric":
        angle_of_next_direction += bat_direction.angle_between(allocentric_axis_y)

    if intensity > parameters_df["SPL_THRESHOLD_FOR_REPULSION"]:
        angle_of_next_direction += np.pi
        response_type = "repulsion_passive"
    elif intensity > parameters_df["SPL_THRESHOLD_FOR_ATTRACTION"]:
        response_type = "attraction_passive"
    else:
        raise ValueError(
            "This condition should never occur as no response is thrown out earlier."
        )

    return bat_direction.rotate(angle_of_next_direction).normalize(), response_type
