import numpy as np
import scipy as scp


def compute_collision_counts_and_length(bat_positions, parameters_df):
    """Calculate collisions given position history of bats

    Args:
        bat_positions (list): 3D array with first index being time, second being agent id
                                and third being (x/y) coordinate.
        parameters_df (dict): parameter file of the simulation run

    Returns:
        int: wall collision counts
        int: bat bat collision counts
        list: wall collision cumulative timeseries
        list: bat bat collision cumulative timerseries
    """
    arena_width = parameters_df["ARENA_WIDTH"]
    arena_length = parameters_df["ARENA_LENGTH"]
    bat_radius = parameters_df["BAT_RADIUS"]

    collision_counter = 0
    wall_collision_counter = 0
    batbat_collision_counter = 0
    track_collision_in_last_frame_w_bats = []
    track_collision_in_last_frame_w_walls = []

    timeseries_cumulative_wall_collision = []
    timeseries_cumulative_bat_bat_collision = []

    for position_frame in bat_positions:
        distance_matrix = scp.spatial.distance_matrix(position_frame, position_frame)
        track_collision_in_current_frame_w_bats = []
        track_collision_in_current_frame_w_walls = []

        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                if i < j:
                    if distance_matrix[i, j] < 2 * bat_radius:
                        track_collision_in_current_frame_w_bats.append((i, j))
                        if (i, j) not in track_collision_in_last_frame_w_bats:
                            collision_counter += 1
                            batbat_collision_counter += 1

        for i, bat in enumerate(position_frame):

            if (
                bat[0] >= arena_width - bat_radius
                or bat[0] <= bat_radius
                or bat[1] >= arena_length - bat_radius
                or bat[1] <= bat_radius
            ):
                track_collision_in_current_frame_w_walls.append(i)

                if i not in track_collision_in_last_frame_w_walls:
                    collision_counter += 1
                    wall_collision_counter += 1

        track_collision_in_last_frame_w_bats = track_collision_in_current_frame_w_bats
        track_collision_in_last_frame_w_walls = track_collision_in_current_frame_w_walls

        timeseries_cumulative_wall_collision.append(wall_collision_counter)
        timeseries_cumulative_bat_bat_collision.append(batbat_collision_counter)

    return (
        wall_collision_counter,
        batbat_collision_counter,
        timeseries_cumulative_wall_collision,
        timeseries_cumulative_bat_bat_collision,
    )
