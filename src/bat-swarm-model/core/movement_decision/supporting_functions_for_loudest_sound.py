import bisect
import sys

import numpy as np

sys.path.append("./src/")
from supporting_files.utilities import str2bool


def find_sum_of_db(list_of_spls):
    """Computes the sum of all the spls in a given list.

    Args:
        list_of_spls (list): contains all the spls (in dB scale) that need to be added

    Returns:
        float: sum of spls in the list (in dB scale)
    """
    _temporary_sum = 0
    for spl in list_of_spls:
        if spl != 0:
            _temporary_sum += 10 ** (spl / 20)
    if _temporary_sum == 0:
        return 0
    sum_of_spls_in_db = 20 * np.log10(_temporary_sum)
    return np.round(sum_of_spls_in_db, 2)


def generate_empty_matrix(parameters_df):
    """generate an empty matrix to store sensory volume information

    Args:
        parameters_df (dict): parameter file used for the simulations

    Returns:
        tuple : first matrix is the empty matrix and
        second is the reference list of angles for columns.
    """
    # radial_resolution = parameters_df["BAT_RADIAL_RESOLUTION"]
    angular_resolution = np.radians(parameters_df["BAT_ANGULAR_RESOLUTION"])
    # post_call_sampling_interval = parameters_df["RESPONSE_DELAY_LOUDEST_SOUND"]

    # spatial_grid_r = np.arange(
    #     call_duration,
    #     post_call_sampling_interval + radial_resolution,
    #     radial_resolution,
    # )
    spatial_grid_theta = np.arange(-np.pi, np.pi, angular_resolution)
    matrix_spatial_grid = np.zeros(shape=(1, len(spatial_grid_theta)))
    return matrix_spatial_grid, spatial_grid_theta


# TODO : stuff copied from the consistency functions file, make a general function instead of copy pasting the same functions


def convert_angles_to_negpi_to_pi(list_of_angles):
    new_list = []
    for angle in list_of_angles:
        if angle > np.pi or angle < -np.pi:
            opposite_sign = np.sign(angle) * -1
            converted_angle = angle % (opposite_sign * np.pi)
        else:
            converted_angle = angle
        new_list.append(converted_angle)
    return np.array(new_list)


def find_indices_corresponding_to_hearing_range(
    parameters_df,
    spatial_grid_theta,
    angle_between_reference_axis_and_bat,
    hearing_range,
):
    bat_angular_resolution = np.radians(parameters_df["BAT_ANGULAR_RESOLUTION"])

    # print(hearing_range, angle_between_reference_axis_and_bat)
    angles_to_hearing_range = np.arange(
        -hearing_range + angle_between_reference_axis_and_bat,
        hearing_range
        + angle_between_reference_axis_and_bat
        + bat_angular_resolution / 2,
        bat_angular_resolution,
    )
    angles_to_hearing_range = convert_angles_to_negpi_to_pi(angles_to_hearing_range)
    indices_corresponding_to_hearing_range = [
        bisect.bisect_left(spatial_grid_theta, i) for i in angles_to_hearing_range
    ]
    indices_corresponding_to_hearing_range = [
        i if i != len(spatial_grid_theta) else 0
        for i in indices_corresponding_to_hearing_range
    ]
    # print(f"angles in hearing range {np.degrees(np.array(angles_to_hearing_range))}")
    return indices_corresponding_to_hearing_range


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

    matrix_based_on_hearing_range = np.zeros(shape=matrix.shape)
    for index in hearing_range_indices:
        matrix_based_on_hearing_range[(0, index)] = matrix[(0, index)]

    return matrix_based_on_hearing_range, np.array(hearing_range_indices)


def convert_detected_sounds_into_grids_loudest_sound(
    heard_sounds, parameters_df, bat_direction, allocentric_axis_y
):
    spatial_reference_frame = parameters_df["SPATIAL_REFERENCE_FRAME"]
    # convert_grid_to_one_hot = str2bool(parameters_df["CONVERT_GRIDS_TO_ONE_HOT"])

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

        grid_column_index = bisect.bisect_left(spatial_grid_theta, theta)

        # angles are circular
        if grid_column_index == len(spatial_grid_theta):
            grid_column_index = 0

        sound_max_intensity = np.max(sound_object["all_spl_values"])

        # if store_grid[0, grid_column_index] >= sound_max_intensity:
        #     continue
        if store_grid[0, grid_column_index] < sound_max_intensity:
            store_grid[0, grid_column_index] = sound_max_intensity

        store_grid_sum[0, grid_column_index] = find_sum_of_db(
            [store_grid_sum[0, grid_column_index], sound_max_intensity]
        )
        # print(f"max db in cells : {store_grid}")
        # print(f"sum of db in cells : {store_grid_sum}")

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
    elif neighbour_selection == "random":

        candidate_cells = np.flatnonzero(matrix > intensity_threshold)

        # if len(candidate_cells)==0"
        # return
        return np.random.choice(candidate_cells)
    else:
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
    # print(matrix)
    angle_between_reference_axis_and_bat = allocentric_axis_y.angle_between(
        bat_direction
    )

    # TODO : implementation to incorporate limited range for attraction and repulsion
    # TODO : add variable range for attraction and repulsion ig? if you dont

    matrix_based_on_attraction_range, attraction_range_indices = (
        modify_matrix_based_on_hearing_range(
            matrix,
            angle_between_reference_axis_and_bat,
            parameters_df,
            angle_threshold=parameters_df["ANGLE_RANGE_FOR_ATTRACTION"],
        )
    )

    matrix_based_on_repulsion_range, repulsion_range_indices = (
        modify_matrix_based_on_hearing_range(
            matrix,
            angle_between_reference_axis_and_bat,
            parameters_df,
            angle_threshold=parameters_df["ANGLE_RANGE_FOR_REPULSION"],
        )
    )

    # previous_output_cell = [0, previous_output_cell_raw[1]]
    # print(matrix_based_on_attraction_range, attraction_range_indices)
    # print(matrix_based_on_repulsion_range, repulsion_range_indices)
    # print(previous_output_cell)
    # print(matrix_based_on_attraction_range)

    intensity_threshold_for_attraction = parameters_df["SPL_THRESHOLD_FOR_ATTRACTION"]
    intensity_threshold_for_repulsion = parameters_df["SPL_THRESHOLD_FOR_REPULSION"]
    neighbour_selection = parameters_df.get("NEIGHBOUR_SELECTION", "random")

    # check if repulsion criteria met?
    # loudest_intensity_in_repulsion_matrix = np.max(matrix_based_on_repulsion_range)
    # print(
    #     [
    #         intensity > intensity_threshold_for_repulsion
    #         for intensity in matrix_based_on_repulsion_range
    #     ]
    # )
    # print(
    #     any(
    #         [
    #             intensity > intensity_threshold_for_repulsion
    #             for intensity in matrix_based_on_repulsion_range
    #         ]
    #     )
    # )
    is_bat_doing_repulsion = any(
        [
            intensity > intensity_threshold_for_repulsion
            for intensity in matrix_based_on_repulsion_range
        ][0]
    )
    # (loudest_intensity_in_repulsion_matrix > intensity_threshold_for_repulsion)

    # loudest_intensity_in_attraction_matrix = np.max(matrix_based_on_attraction_range)
    is_bat_doing_attraction = any(
        [
            intensity > intensity_threshold_for_attraction
            for intensity in matrix_based_on_attraction_range
        ][0]
    )
    # print(is_bat_doing_repulsion)
    if is_bat_doing_repulsion:
        # TODO : add sustainance of response
        if (
            previous_output_cell[1] in repulsion_range_indices
            and matrix_based_on_repulsion_range[0, previous_output_cell[1]]
            > intensity_threshold_for_repulsion
        ):
            return (0, previous_output_cell[1])

        selected_cell_in_repulsion_matrix = select_cell_to_respond_to(
            matrix_based_on_repulsion_range,
            intensity_threshold_for_repulsion,
            neighbour_selection,
        )
        output_cell_number = (0, selected_cell_in_repulsion_matrix)
    elif is_bat_doing_attraction:
        # TODO : add sustainance of response
        if (
            previous_output_cell[1] in attraction_range_indices
            and matrix_based_on_attraction_range[0, previous_output_cell[1]]
            > intensity_threshold_for_attraction
        ):
            return (0, previous_output_cell[1])

        selected_cell_in_attraction_matrix = select_cell_to_respond_to(
            matrix_based_on_repulsion_range,
            intensity_threshold_for_attraction,
            neighbour_selection,
        )
        output_cell_number = (0, selected_cell_in_attraction_matrix)
    elif not (is_bat_doing_repulsion or is_bat_doing_attraction):
        output_cell_number = (np.nan, np.nan)
    else:
        print(is_bat_doing_repulsion, is_bat_doing_attraction)
        raise ValueError(
            "the agent is attracting and repelling at the same time! impossible!!!"
        )
    # loudest_cell = np.argmax(matrix)
    # output_cell_number = (0, loudest_cell)

    # print(matrix[output_cell_number])
    # if matrix[output_cell_number] < intensity_threshold_for_attraction:
    #     output_cell_number = [np.nan, np.nan]

    # if id == 0:
    #     print(
    #         f"attraction? {is_bat_doing_attraction} repulsion? {is_bat_doing_repulsion}"
    #     )
    #     print(
    #         f"loudest intensity attract {loudest_intensity_in_attraction_matrix}, repel {loudest_intensity_in_repulsion_matrix}"
    #     )
    #     print(f"output {output_cell_number}")

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
    intensity_threshold_for_repulsion = parameters_df["SPL_THRESHOLD_FOR_REPULSION"]
    intensity_threshold_for_attraction = parameters_df["SPL_THRESHOLD_FOR_ATTRACTION"]

    if spatial_reference_frame == "allocentric":
        angle_between_self_and_allocentric_axis = bat_direction.angle_between(
            allocentric_axis_y
        )
        angle_of_next_direction += angle_between_self_and_allocentric_axis

    # print(intensity)
    if intensity > intensity_threshold_for_repulsion:
        angle_of_next_direction += np.pi
        response_type = "repulsion_passive"
    elif intensity > intensity_threshold_for_attraction:
        # print(f"attraction delay{corrected_time_delay}")
        # angle_of_next_direction += np.pi / 2
        response_type = "attraction_passive"
    else:
        raise ValueError(
            "This condition should never occur as no response is thrown out earlier."
        )

    next_direction = bat_direction.rotate(angle_of_next_direction)

    return next_direction.normalize(), response_type


if __name__ == "__main__":
    dict_params = {
        "BAT_ANGULAR_RESOLUTION": 30,
        "CALL_DURATION": 2,
    }
    X, Y = generate_empty_matrix(parameters_df=dict_params)
    Z = np.matrix([[0, 1, 2, 1]])
    print(np.argmin(X))
    print(np.argmax(Z), Z[0, np.argmax(Z)])
