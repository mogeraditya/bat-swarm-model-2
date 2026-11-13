"""The big 2D arena view: bats, obstacles, jammers, trajectories and sounds.

This is exactly the `ax[0, 1]` panel from the original `setup_visualization`
/ `visualize` functions, just packaged as a module.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle, Rectangle, Wedge
from plotting.base_module import AnimationModule


class ArenaModule(AnimationModule):
    GRID_POS = (0, 1)
    PROJECTION = None

    def setup(
        self,
        parameters_df,
        bats,
        obstacles,
        jammers,
        history,
        focal_bat,
        show_sounds,
        **_,
    ):
        self.parameters_df = parameters_df
        self.bats = bats
        self.show_sounds = show_sounds
        self.focal_bat = focal_bat

        ax = self.ax

        print(parameters_df["ARENA_WIDTH"])
        ax.set_xlim(0, parameters_df["ARENA_WIDTH"])
        ax.set_ylim(0, parameters_df["ARENA_LENGTH"])
        ax.set_aspect("equal")
        ax.set_title("Bat Echolocation with Direct Calls and Echoes")

        boundary = Rectangle(
            (0, 0),
            parameters_df["ARENA_WIDTH"],
            parameters_df["ARENA_LENGTH"],
            fill=False,
            linestyle="--",
            color="gray",
        )
        ax.add_patch(boundary)

        self.obstacle_patches = []
        for obstacle in obstacles:
            obs_circle = Circle(
                (obstacle.position.x, obstacle.position.y),
                obstacle.radius,
                color="red",
                alpha=0.5,
            )
            ax.add_patch(obs_circle)
            self.obstacle_patches.append(obs_circle)

        self.jammer_patches = []
        for jammer in jammers:
            direction_arrow = Arrow(
                jammer.position.x,
                jammer.position.y,
                jammer.direction.x,
                jammer.direction.y,  # Initial direction (0, 0)
                width=0.3,  # adjust arrow width as needed here
                color="red",
                alpha=0.8,
            )
            jam_circle = Circle(
                (jammer.position.x, jammer.position.y),
                jammer.radius,
                color="red",
                alpha=0.5,
            )
            ax.add_patch(jam_circle)
            ax.add_patch(direction_arrow)
            self.jammer_patches.append(jam_circle)

        self.bat_markers = []
        self.direction_arrows = []
        self.next_direction_arrows = []
        self.sound_artists = []
        self.detection_artists = []
        self.trajectory_lines = []
        self.trajectory_history = [[] for _ in range(len(bats))]
        num_colors = len(bats)

        cm = plt.get_cmap("gist_rainbow")
        self.cm = cm

        for i, bat in enumerate(bats):
            color_cm = cm(i / num_colors) if i != 0 else "red"
            # trajectory line place holders
            (trajectory_line,) = ax.plot(
                [],
                [],
                color=color_cm,
                linestyle="--",
                alpha=1,
                linewidth=1.5,
            )
            self.trajectory_lines.append(trajectory_line)

            # bat object placeholders
            bat_circle = Circle(
                (bat.position.x, bat.position.y),
                bat.radius,
                color=cm(i / num_colors),
                label=f"Bat {bat.id}",
            )
            ax.add_patch(bat_circle)

            self.bat_markers.append(bat_circle)

            # direction arrow place holders
            direction_arrow = Arrow(
                bat.position.x,
                bat.position.y,
                0,
                0,  # Initial direction (0, 0)
                width=0.3,  # adjust arrow width as needed here
                color=cm(i / num_colors),
                alpha=0.8,
            )

            next_direction_arrow = Arrow(
                bat.position.x,
                bat.position.y,
                0,
                0,  # Initial direction (0, 0)
                width=0.3,  # adjust arrow width as needed here
                color="white",
                ec=cm(i / num_colors),
                alpha=0.8,
            )

            ax.add_patch(direction_arrow)
            self.direction_arrows.append(direction_arrow)
            ax.add_patch(next_direction_arrow)
            self.next_direction_arrows.append(next_direction_arrow)
        print(self.bat_markers)
        return (
            self.bat_markers
            + self.direction_arrows
            + self.next_direction_arrows
            + self.trajectory_lines
        )

    def init_frame(self):
        for marker in self.bat_markers:

            marker.center = (np.nan, np.nan)
        for arrow in self.direction_arrows:
            arrow.set_data(x=np.nan, y=np.nan, dx=0, dy=0)
        for arrow in self.next_direction_arrows:
            arrow.set_data(x=np.nan, y=np.nan, dx=0, dy=0)
        for line in self.trajectory_lines:
            line.set_data([], [])

        return self.bat_markers + self.direction_arrows + self.trajectory_lines

    def update_frame(self, i, frame, context):
        num_colors = len(self.bats)
        cm = self.cm
        ax = self.ax

        for j, (x, y) in enumerate(frame["bat_positions"]):
            self.bat_markers[j].center = (x, y)
            # print(f"framenumber {i}, bat {j}, bat_position {(x,y)}")
            if "bat_directions" in frame and j < len(frame["bat_directions"]):
                dx, dy = frame["bat_directions"][j]
                # Scale the direction vector for better visualization
                scale = 0.5  # Adjust this scale factor as needed
                self.direction_arrows[j].set_data(
                    x=x, y=y, dx=dx * scale, dy=dy * scale
                )
            if "bat_response_vector" in frame and j < len(frame["bat_response_vector"]):
                dx, dy = frame["bat_response_vector"][j]
                # Scale the direction vector for better visualization
                scale = 0.5  # Adjust this scale factor as needed
                if frame["response_type"][j] == "repulsion":
                    # ls = "--"
                    # ec = "red"
                    fcolor = "black"
                elif frame["response_type"][j] == "attraction":
                    fcolor = "white"
                elif frame["response_type"][j] == "attraction_passive":
                    fcolor = "cyan"
                elif frame["response_type"][j] == "repulsion_passive":
                    fcolor = "purple"
                else:
                    fcolor = cm(j / num_colors)
                self.next_direction_arrows[j].set_data(
                    x=x, y=y, dx=dx * scale, dy=dy * scale
                )
                self.next_direction_arrows[j].set(fc=fcolor)
            # if j == focal_bat:
            self.trajectory_history[j].append((x, y))

            # Keep only the last 400 positions
            if len(self.trajectory_history[j]) > 300:
                self.trajectory_history[j].pop(0)

            # Update trajectory line
            if len(self.trajectory_history[j]) > 300:
                x_vals, y_vals = (
                    np.array(self.trajectory_history[j])[:, 0],
                    np.array(self.trajectory_history[j])[:, 1],
                )
                self.trajectory_lines[j].set_data(x_vals, y_vals)
            # print(self.trajectory_history)
        for artist in self.sound_artists + self.detection_artists:
            artist.remove()

        self.sound_artists.clear()
        self.detection_artists.clear()
        ax.set_title(f"time step: {frame["time"]:.5f}")

        if self.show_sounds:
            for sound in frame["sound_objects"]:
                if not sound["status"]:
                    continue

                emitter_color = cm(sound["emitter_id"] / num_colors)
                alpha = 0.5 - (0.1 * sound.get("reflection_count", 0))

                inner = max(
                    0,
                    sound["radius"]
                    - self.parameters_df["CALL_DURATION"]
                    * self.parameters_df["SOUND_SPEED"],
                )
                outer = sound["radius"]

                if inner < outer:
                    if sound["type"] == "direct":
                        linestyle = "-"
                        hatching_of_disk = "++"
                    else:
                        linestyle = "--"
                        alpha = 0.5 * alpha
                        hatching_of_disk = ".."
                    if inner == 0:
                        width_of_disk = sound["radius"]
                    else:
                        width_of_disk = (
                            self.parameters_df["CALL_DURATION"]
                            * self.parameters_df["SOUND_SPEED"]
                        )
                    wedge = Wedge(
                        sound["origin"],
                        outer,
                        0,
                        360,
                        width=width_of_disk,
                        fill=False,
                        color=emitter_color,
                        alpha=alpha,
                        linestyle=linestyle,
                        hatch=hatching_of_disk,
                    )
                    ax.add_patch(wedge)
                    self.sound_artists.append(wedge)

        return (
            self.bat_markers
            + self.direction_arrows
            + self.next_direction_arrows
            + self.trajectory_lines
            + self.sound_artists
            + self.detection_artists
        )

    def get_legend_handles(self):
        return self.ax.get_legend_handles_labels()
