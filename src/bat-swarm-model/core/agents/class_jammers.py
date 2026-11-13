import random

import numpy as np
from agents.class_sounds import DirectSound
from supporting_files.vectors import Vector


class Jammers:
    _id_counter = 0

    def __init__(self, parameters_df, position, direction, call_rate, wall_id=None):
        self.id = Jammers._id_counter
        Jammers._id_counter += 1

        self.parameters_df = parameters_df
        self.position = position
        self.direction = direction.normalize()
        self.radius = 0.125

        time_step_size = self.parameters_df["TIME_STEP"]
        # find the number of decimal places to set rounding equal to time step size
        self.rounding_based_on_time_step = len(str(time_step_size).split(".")[1])

        self.time_since_last_call = -np.inf
        self.call_rate = call_rate
        self.time_since_last_call = np.round(
            random.uniform(0, 1 / self.call_rate),
            self.rounding_based_on_time_step,
        )
        self.emit_times = []

        # TODO : special directivity for jammer speakers.
        # TODO : think about wall reflection implementation.

        self.wall_id = wall_id if wall_id is not None else None

    def update(self, current_time, sound_objects):
        """Function to update jammerss with time.
        This function handles sound emission.

        Args:
            current_time (float): Time, in seconds, for which the simualtion has been running.
            sound_objects (EchoSound): direct and echo sounds that are currently active.
        """

        self.emit_sounds(current_time, sound_objects)

    def emit_sounds(self, current_time, sound_objects):
        """Trigger sound emission by Jammer.
        Whenever the function is called, it checks if sufficient time
        has passed and a DirectSoundObject is created.

        Args:
            current_time (float): Time, in seconds, for which the simualtion has been running.
            sound_objects (list): List containing all active sounds in the simulation
        """
        self.time_since_last_call += self.parameters_df["TIME_STEP"]
        call_interval = 1.0 / self.call_rate

        if self.time_since_last_call >= call_interval:
            sound = DirectSound(
                parameters_df=self.parameters_df,
                origin=self.position,
                creation_time=current_time,
                emitter_id=self.id,
                direction_vector=self.direction,
            )
            sound.wall_id = self.wall_id
            self.emit_times.append(current_time)
            sound_objects.append(sound)

            self.time_since_last_call = np.random.uniform(
                -self.parameters_df["NOISE_IN_CALL_RATE"],
                self.parameters_df["NOISE_IN_CALL_RATE"],
            )

    def __repr__(self):
        return f"Jammer(id={self.id}, position={self.position}, direction={self.direction})"


def make_jammers(parameters_df):

    jammer_resolution = parameters_df["JAMMER_RESOLUTION"]
    call_rate = parameters_df["CALL_RATE"]
    width_array = np.arange(
        jammer_resolution,
        parameters_df["ARENA_WIDTH"],
        jammer_resolution,
    )
    length_array = np.arange(
        jammer_resolution,
        parameters_df["ARENA_LENGTH"],
        jammer_resolution,
    )

    left_wall = [Vector(0, i) for i in length_array]
    right_wall = [Vector(parameters_df["ARENA_WIDTH"], i) for i in length_array]
    top_wall = [Vector(i, parameters_df["ARENA_LENGTH"]) for i in width_array]
    bottom_wall = [Vector(i, 0) for i in width_array]

    left_jammer_directions = [Vector(1, 0) for i in left_wall]
    right_jammer_directions = [Vector(-1, 0) for i in right_wall]
    top_jammer_directions = [Vector(0, -1) for i in top_wall]
    bottom_jammer_directions = [Vector(0, 1) for i in bottom_wall]

    positions_to_put_jammers = [*left_wall, *right_wall, *top_wall, *bottom_wall]
    directions_of_jammers = [
        *left_jammer_directions,
        *right_jammer_directions,
        *top_jammer_directions,
        *bottom_jammer_directions,
    ]
    wall_ids = [
        *["left_wall"] * len(left_wall),
        *["right_wall"] * len(right_wall),
        *["top_wall"] * len(top_wall),
        *["bottom_wall"] * len(bottom_wall),
    ]

    store_wall_objects = [
        Jammers(
            parameters_df, position, directions_of_jammers[i], call_rate, wall_ids[i]
        )
        for i, position in enumerate(positions_to_put_jammers)
    ]
    return store_wall_objects
