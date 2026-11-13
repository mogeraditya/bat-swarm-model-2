"""The two figtext counters (ipi number / call time). No dedicated axis."""

import matplotlib.pyplot as plt
import numpy as np
from plotting.base_module import AnimationModule


class InfoTextModule(AnimationModule):
    GRID_POS = None  # doesn't need its own subplot
    PROJECTION = None

    def add_axis(self, fig, gridspec):
        # text is drawn directly onto the figure, no axis needed.
        self.fig = fig
        self.ax = None
        return None

    def setup(self, focal_bat, **_):
        self.focal_bat = focal_bat

        self.ipi_counter_plot = plt.figtext(
            0.01,
            0.8,
            f"IPI number : {0}",
            fontsize=25,
            ha="left",
            va="top",
        )
        self.call_time_plot = plt.figtext(
            0.01,
            0.9,
            f"call time : {-np.inf}",
            fontsize=25,
            ha="left",
            va="top",
        )
        return [self.ipi_counter_plot, self.call_time_plot]

    def init_frame(self):
        return [self.ipi_counter_plot, self.call_time_plot]

    def update_frame(self, i, frame, context):
        self.ipi_counter_plot.set_text(
            f"IPI number : {frame["bat_ipi_counters"][self.focal_bat]}"
        )
        self.call_time_plot.set_text(
            f"call time : {frame["bat_call_time"][self.focal_bat]}"
        )
        return [self.ipi_counter_plot, self.call_time_plot]
