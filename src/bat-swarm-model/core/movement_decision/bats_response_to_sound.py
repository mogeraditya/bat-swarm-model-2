import numpy as np
from movement_decision.supporting_functions_for_consistency_2 import (  # given_matrix_find_cell_to_respond_to_absence,
    convert_detected_sounds_into_grids,
    given_matrix_find_cell_to_respond_to_presence,
    given_time_and_angle_return_direction,
)
from movement_decision.supporting_functions_for_loudest_sound import (
    convert_detected_sounds_into_grids_loudest_sound,
    given_matrix_find_cell_to_respond_to_loudest_sound,
    given_time_and_angle_return_direction_loudest_sound,
)
from supporting_files.vectors import Vector


def find_sum_of_db_matrices(list_of_matrices):
    """Return elementwise dB sum of equally shaped matrices; zeros are ignored."""
    linear_sum = np.sum(
        [10 ** (matrix / 20) * (matrix != 0) for matrix in list_of_matrices], axis=0
    )
    output_matrix = np.zeros_like(linear_sum, dtype=float)
    mask = linear_sum != 0
    output_matrix[mask] = np.round(20 * np.log10(linear_sum[mask]), 2)
    return output_matrix


def decide_next_direction_based_on_consistency(self, detected_sound_objects):
    """decide next direction of bat based on sound

    Args:
        detected_sound_objects (list): list containing detected sounds
    """
    len_cap_memory_window = self.parameters_df["MEMORY_WINDOW_FOR_CONSISTENCY"]
    number_of_consistent_ipis_for_behaviour = int(
        np.ceil(self.parameters_df["MEMORY_WINDOW_FOR_CONSISTENCY"] / 2)
    )
    (
        grid_for_current_ipi,
        grid_for_current_ipi_intensity,
        grid_row_labels,
        grid_column_labels,
    ) = convert_detected_sounds_into_grids(
        detected_sound_objects,
        self.parameters_df,
        self.direction,
        self.allocentric_axis_y,
    )

    self.memory_window_to_store_grids.append(grid_for_current_ipi)
    self.memory_window_to_store_grids_intensity.append(grid_for_current_ipi_intensity)

    length_of_memory_window = len(self.memory_window_to_store_grids)

    if length_of_memory_window > len_cap_memory_window:
        self.memory_window_to_store_grids.pop(0)
        self.memory_window_to_store_grids_intensity.pop(0)
    elif length_of_memory_window <= len_cap_memory_window:
        number_of_consistent_ipis_for_behaviour = np.min(
            [number_of_consistent_ipis_for_behaviour, length_of_memory_window - 1]
        )

    grids_to_consider_for_direction_change = self.memory_window_to_store_grids.copy()
    sum_grids_in_memory = np.sum(grids_to_consider_for_direction_change, axis=0)
    sum_grids_in_memory_intensity = find_sum_of_db_matrices(
        self.memory_window_to_store_grids_intensity
    )

    self.memory_window_sum_matrix = sum_grids_in_memory.copy()
    self.sum_grids_in_memory_intensity = sum_grids_in_memory_intensity.copy()
    self.ipi_matrix_for_consistency = grids_to_consider_for_direction_change[-1]

    cell_index_to_respond_to = given_matrix_find_cell_to_respond_to_presence(
        sum_grids_in_memory,
        sum_grids_in_memory_intensity,
        number_of_consistent_ipis_for_behaviour,
        self.parameters_df,
        self.direction,
        self.allocentric_axis_y,
        self.cell_index_to_respond_to,
    )

    self.cell_index_to_respond_to = cell_index_to_respond_to
    if np.isnan(cell_index_to_respond_to[0]):
        next_direction = self.next_direction
        response_type = None
        self.any_consistent_sound = "no"
        time_delay_of_activated_cell = np.nan

    else:
        time_delay_of_activated_cell = grid_row_labels[cell_index_to_respond_to[0]]
        angle_of_activated_cell = grid_column_labels[cell_index_to_respond_to[1]]
        next_direction, response_type = given_time_and_angle_return_direction(
            angle_of_activated_cell,
            self.parameters_df,
            self.direction,
            self.allocentric_axis_y,
        )
        self.any_consistent_sound = "yes"

    # # TODO : min delay implementation for slowing down
    # min_delay = np.inf
    # for sound_object in detected_sound_objects:
    #     delta_t = (
    #         np.array(sound_object["occurance_times"])[0]
    #         - sound_object["bat_last_call_time"]
    #     )
    #     min_delay = min(min_delay, delta_t)
    #     print(min_delay, delta_t)
    # if min_delay == np.inf:
    #     min_delay = np.nan
    self.first_detection_time_delay = time_delay_of_activated_cell

    return next_direction.normalize(), response_type


def decide_next_direction_based_on_loudness(self, detected_sound_objects):
    """decide next direction of bat based on sound

    Args:
        detected_sound_objects (list): list containing detected sounds
    """

    self.ipi_matrix_for_sensory_volume, grid_column_labels = (
        convert_detected_sounds_into_grids_loudest_sound(
            detected_sound_objects,
            self.parameters_df,
            self.direction,
            self.allocentric_axis_y,
        )
    )

    cell_index_to_respond_to = given_matrix_find_cell_to_respond_to_loudest_sound(
        self.ipi_matrix_for_sensory_volume,
        self.parameters_df,
        self.direction,
        self.allocentric_axis_y,
        self.id,
        self.cell_index_to_respond_to,
    )

    self.cell_index_to_respond_to = cell_index_to_respond_to

    if np.isnan(cell_index_to_respond_to[0]):
        next_direction = self.next_direction
        response_type = None
        self.any_loud_sound = "no"

    else:
        intensity_of_sound = self.ipi_matrix_for_sensory_volume[
            cell_index_to_respond_to
        ]
        angle_of_activated_cell = grid_column_labels[cell_index_to_respond_to[1]]
        next_direction, response_type = (
            given_time_and_angle_return_direction_loudest_sound(
                intensity_of_sound,
                angle_of_activated_cell,
                self.parameters_df,
                self.direction,
                self.allocentric_axis_y,
            )
        )
        self.any_loud_sound = "yes"
        # print(next_direction.normalize(), angle_of_activated_cell, intensity_of_sound)

    self.ipi_matrix_for_sensory_volume = self.ipi_matrix_for_sensory_volume[0]
    return next_direction.normalize(), response_type
