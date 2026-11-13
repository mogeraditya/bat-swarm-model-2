"""'sensory_volume' polar line/fill plot (originally ax[1, 2])."""

import sys

import numpy as np
from plotting.base_module import AnimationModule
from plotting.common import (
    increase_sensory_volume_resolution_for_plotting,
    plot_sensory_volume,
)

sys.path.append("./src/bat-swarm-model/core/")
from movement_decision.supporting_functions_for_consistency import (
    given_parameters_df_return_grid_matrix_zeros,
)


class SensoryVolumeModule(AnimationModule):
    GRID_POS = (1, 2)
    PROJECTION = "polar"

    def setup(self, parameters_df, history, focal_bat, **_):
        ax = self.ax

        rows, columns = given_parameters_df_return_grid_matrix_zeros(parameters_df)[1:3]
        spatial_grid_theta = columns

        sensory_volume_data_time_series = [
            frame["bat_ipi_matrix_for_sensory_volume"][focal_bat] for frame in history
        ]
        spatial_grid_theta_finer = increase_sensory_volume_resolution_for_plotting(
            sensory_volume_data_time_series[0], spatial_grid_theta, 10
        )[1]
        sensory_volume_data_time_series = [
            increase_sensory_volume_resolution_for_plotting(i, spatial_grid_theta, 10)[
                0
            ]
            for i in sensory_volume_data_time_series
        ]

        self.spatial_grid_theta_finer = spatial_grid_theta_finer
        self.sensory_volume_data_time_series = sensory_volume_data_time_series

        ax.set_title("sensory_volume")
        ax.tick_params("y", rotation=30)

        self.im2, self.fill = plot_sensory_volume(
            sensory_volume_data_time_series[0],
            spatial_grid_theta_finer,
            fig_and_ax=(self.fig, ax),
        )

        return list(self.im2) + list(self.fill)

    def init_frame(self):
        return list(self.im2) + list(self.fill)

    def update_frame(self, i, frame, context):
        self.im2[0].set_data(
            self.spatial_grid_theta_finer, self.sensory_volume_data_time_series[i]
        )
        self.fill[0].set_xy(
            np.column_stack(
                [self.spatial_grid_theta_finer, self.sensory_volume_data_time_series[i]]
            )
        )
        return list(self.im2) + list(self.fill)
