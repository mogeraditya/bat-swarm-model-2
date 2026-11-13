"""'responses in 5 ipi' radial heat-map (originally ax[1, 1])."""

import sys

import numpy as np
from plotting.base_module import AnimationModule
from plotting.common import plot_grid_matrix_into_radial_time_series
from plotting.convert_heard_sounds_to_matrix import convert_matrix_for_plotting_nicer

sys.path.append("./src/bat-swarm-model/core/")
from movement_decision.supporting_functions_for_consistency import (
    given_parameters_df_return_grid_matrix_zeros,
)


class SumRadialModule(AnimationModule):
    GRID_POS = (1, 1)
    PROJECTION = "polar"

    def setup(self, parameters_df, history, focal_bat, **_):
        ax = self.ax
        grid_data_time_series_sum = [frame["bat_sum_matrix"] for frame in history]
        spatial_grid_r, spatial_grid_theta = (
            given_parameters_df_return_grid_matrix_zeros(parameters_df)[1:3]
        )

        converted_grids = [
            convert_matrix_for_plotting_nicer(
                grid, spatial_grid_r, spatial_grid_theta, 10, focal_bat
            )
            for grid in grid_data_time_series_sum
        ]

        self.new_grid_data_time_series_sum = np.asarray(
            [converted_grid[0] for converted_grid in converted_grids]
        )
        new_angular_resolution = converted_grids[0][1]

        # Each converted grid may be a list of bat matrices: [focal_bat_matrix].
        first_grid = np.asarray(self.new_grid_data_time_series_sum[0, 0])
        number_of_angular_cells = first_grid.shape[-1]

        new_spatial_grid_theta = np.linspace(
            -np.pi + new_angular_resolution / 2,
            np.pi - new_angular_resolution / 2,
            number_of_angular_cells,
        )

        ax.set_title("responses in 5 ipi")
        ax.tick_params("y", rotation=30)

        self.im = plot_grid_matrix_into_radial_time_series(
            first_grid,
            spatial_grid_r,
            new_spatial_grid_theta,
            fig_and_ax=(self.fig, ax),
            cmin=0,
            cmax=5,
        )
        return [self.im]

    def init_frame(self):
        return [self.im]

    def update_frame(self, i, frame, context):
        self.im.set_array(np.asarray(self.new_grid_data_time_series_sum[i, 0]).T)
        return [self.im]
