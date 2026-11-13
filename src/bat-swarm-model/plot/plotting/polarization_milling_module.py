"""Polarization & milling score line plot (position 0, 2).

Plots a rolling 1-second average of `polarization_score` and `milling_score`
(present in each frame's data dict). Both lines update every frame, like all
other animated panels, growing to show the full history up to the current
frame.
"""

import bisect

import matplotlib.pyplot as plt
from plotting.base_module import AnimationModule


class PolarizationMillingModule(AnimationModule):
    GRID_POS = (0, 2)
    PROJECTION = None

    def setup(self, history, **_):
        ax = self.ax

        self.times = [frame["time"] for frame in history]
        polarization_raw = [frame["polarization_score"] for frame in history]
        milling_raw = [frame["milling_score"] for frame in history]
        local_polarization_raw = [
            frame["local_polarization_score"] for frame in history
        ]
        # Pre-compute rolling 1-second average for every frame in the history.
        # `self.times` is sorted ascending (guaranteed by stitch_together_history_lists),
        # so we can use bisect to find the start of the 1-second window efficiently.
        self.avg_polarization = []
        self.avg_milling = []
        self.avg_local_polarization = []
        window_seconds = 1.0

        for i, t in enumerate(self.times):
            start_idx = bisect.bisect_left(self.times, t - window_seconds)
            window_polarization = polarization_raw[start_idx : i + 1]
            window_milling = milling_raw[start_idx : i + 1]
            window_local_polarization = local_polarization_raw[start_idx : i + 1]
            self.avg_polarization.append(
                sum(window_polarization) / len(window_polarization)
            )
            self.avg_milling.append(sum(window_milling) / len(window_milling))
            self.avg_local_polarization.append(
                sum(window_local_polarization) / len(window_local_polarization)
            )

        ax.set_title("Polarization & Milling Score (1s avg)")
        ax.set_xlim(self.times[0], self.times[-1])
        ax.set_ylim(0, 1)

        (self.polarization_line,) = ax.plot(
            [], [], color="cyan", linewidth=1.5, label="Polarization"
        )
        (self.milling_line,) = ax.plot(
            [], [], color="magenta", linewidth=1.5, label="Milling"
        )
        (self.local_polarization_line,) = ax.plot(
            [], [], color="yellow", linewidth=1.5, label="Local Polarization"
        )
        ax.legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.3,
            facecolor="black",
            edgecolor="white",
        )

        return [self.polarization_line, self.milling_line, self.local_polarization_line]

    def init_frame(self):
        self.polarization_line.set_data([], [])
        self.milling_line.set_data([], [])
        self.local_polarization_line.set_data([], [])
        return [self.polarization_line, self.milling_line, self.local_polarization_line]

    def update_frame(self, i, frame, context):
        self.polarization_line.set_data(
            self.times[: i + 1], self.avg_polarization[: i + 1]
        )
        self.milling_line.set_data(self.times[: i + 1], self.avg_milling[: i + 1])
        self.local_polarization_line.set_data(
            self.times[: i + 1], self.avg_local_polarization[: i + 1]
        )
        return [self.polarization_line, self.milling_line, self.local_polarization_line]

    # def setup(self, history, **_):
    #     ax = self.ax

    #     self.times = [frame["time"] for frame in history]
    #     polarization_raw = [frame["polarization_score"] for frame in history]
    #     milling_raw = [frame["milling_score"] for frame in history]
    #     local_polarization_raw = [
    #         frame["local_polarization_score_list"] for frame in history
    #     ]

    #     # Pre-compute rolling 1-second average and spread (min/max) for local polarization
    #     self.avg_local_polarization = []
    #     self.min_local_polarization = []
    #     self.max_local_polarization = []
    #     window_seconds = 1.0

    #     for i, t in enumerate(self.times):
    #         start_idx = bisect.bisect_left(self.times, t - window_seconds)
    #         window_local_polarization = local_polarization_raw[start_idx : i + 1]
    #         if window_local_polarization:
    #             print(window_local_polarization)
    #             avg = np.sum(window_local_polarization) / len(window_local_polarization)
    #             min_val = np.min(window_local_polarization)
    #             max_val = np.max(window_local_polarization)
    #         else:
    #             avg = min_val = max_val = 0.0

    #         self.avg_local_polarization.append(avg)
    #         self.min_local_polarization.append(min_val)
    #         self.max_local_polarization.append(max_val)

    #     # Pre-compute rolling averages for polarization and milling (unchanged)
    #     self.avg_polarization = []
    #     self.avg_milling = []
    #     for i, t in enumerate(self.times):
    #         start_idx = bisect.bisect_left(self.times, t - window_seconds)
    #         window_polarization = polarization_raw[start_idx : i + 1]
    #         window_milling = milling_raw[start_idx : i + 1]
    #         self.avg_polarization.append(
    #             sum(window_polarization) / len(window_polarization)
    #             if window_polarization
    #             else 0
    #         )
    #         self.avg_milling.append(
    #             sum(window_milling) / len(window_milling) if window_milling else 0
    #         )

    #     ax.set_title(
    #         "Polarization & Milling Score (1s avg) + Local Polarization Spread"
    #     )
    #     ax.set_xlim(self.times[0], self.times[-1])
    #     ax.set_ylim(0, 1)

    #     # Plot lines for polarization and milling (unchanged)
    #     (self.polarization_line,) = ax.plot(
    #         [], [], color="cyan", linewidth=1.5, label="Polarization"
    #     )
    #     (self.milling_line,) = ax.plot(
    #         [], [], color="magenta", linewidth=1.5, label="Milling"
    #     )

    #     # For local polarization: line + shaded spread region
    #     (self.local_polarization_line,) = ax.plot(
    #         [], [], color="yellow", linewidth=1.5, label="Local Polarization (mean)"
    #     )
    #     self.local_polarization_fill = ax.fill_between(
    #         [], [], [], color="yellow", alpha=0.3, label="Local Polarization Spread"
    #     )

    #     ax.legend(
    #         loc="upper right",
    #         fontsize=9,
    #         framealpha=0.3,
    #         facecolor="black",
    #         edgecolor="white",
    #     )

    #     return [
    #         self.polarization_line,
    #         self.milling_line,
    #         self.local_polarization_line,
    #         self.local_polarization_fill,
    #     ]

    # def init_frame(self):
    #     self.polarization_line.set_data([], [])
    #     self.milling_line.set_data([], [])
    #     self.local_polarization_line.set_data([], [])
    #     self.local_polarization_fill.set_xy([])  # Clear fill
    #     return [
    #         self.polarization_line,
    #         self.milling_line,
    #         self.local_polarization_line,
    #         self.local_polarization_fill,
    #     ]

    # def update_frame(self, i, frame, context):
    #     # Update polarization and milling (unchanged)
    #     self.polarization_line.set_data(
    #         self.times[: i + 1], self.avg_polarization[: i + 1]
    #     )
    #     self.milling_line.set_data(self.times[: i + 1], self.avg_milling[: i + 1])

    #     # Update local polarization: mean line + shaded spread
    #     t_slice = self.times[: i + 1]
    #     avg_slice = self.avg_local_polarization[: i + 1]
    #     min_slice = self.min_local_polarization[: i + 1]
    #     max_slice = self.max_local_polarization[: i + 1]

    #     self.local_polarization_line.set_data(t_slice, avg_slice)

    #     # Update fill_between: fill between min and max
    #     self.local_polarization_fill.set_xy(
    #         list(zip(t_slice, min_slice))
    #         + list(zip(reversed(t_slice), reversed(max_slice)))
    #     )

    #     return [
    #         self.polarization_line,
    #         self.milling_line,
    #         self.local_polarization_line,
    #         self.local_polarization_fill,
    #     ]
