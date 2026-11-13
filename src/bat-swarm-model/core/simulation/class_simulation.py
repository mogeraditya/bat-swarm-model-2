"""Contains the code that describes a Simulation object. Runs one instance, given parameters."""

import json
import os
import pickle
import sys
import uuid
from datetime import datetime

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# from tqdm.rich import tqdm
from tqdm.auto import tqdm

# from supporting_files.store_history import CompactHistoryManager

sys.path.append("./src/")
sys.path.append("./scripts/")

from agents.class_bats import Bat
from agents.class_jammers import make_jammers
from agents.class_obstacles import Obstacle
from agents.class_sounds import DirectSound
from agents.make_walls import make_walls
from data_storage.handle_data_storage import handle_data_storage
from scores.local_polarization_score import (
    aggregate_local_polarization,
    compute_local_polarization_per_bat,
)
from scores.milling_score import compute_milling
from scores.polarization_score import compute_polarization
from supporting_files.handle_reflections import handle_reflections
from supporting_files.utilities import (
    creation_time_calculation,
    load_parameters,
    read_temporal_masking_fn,
)
from supporting_files.vectors import Vector

_X_AXIS = Vector(1, 0)


def _as_np_float(value):
    """History values were np.float64 when Vector used np.round; keep that type
    in the stored output (Vector components are now plain floats)."""
    return np.float64(value) if type(value) is float else value


# @hydra.main(version_base=None, config_path="../config/", config_name="config")
def cfg_to_flat_dict(cfg):
    # print(cfg)
    nested = OmegaConf.to_container(cfg, resolve=True)
    # print(nested)
    flat = {}
    for group in nested.values():
        flat.update(group)
    # print(flat)
    return flat


class Simulation:
    """one instance of the simulation;
    this object's goal is to run the simulation for one
    instance of the set of parameters chosen
    """

    def __init__(
        self,
        cfg,
        output_dir,
        store_history=True,
        store_data_for_plotting=True,
        use_hydra=True,
        termination_criteria="time",
    ):
        if use_hydra:
            parameters_df = cfg_to_flat_dict(cfg)
        else:
            parameters_df = cfg
        # print(parameters_df)
        # print(f"simulation id {id(self)}")
        # parameters_df = cfg
        Bat._id_counter = 0
        Obstacle._id_counter = 0
        self.parameters_df = parameters_df
        self.output_dir = output_dir + "/"
        self.dir_to_store = self.output_dir  # + "/data_for_plotting/"

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dir_to_store, exist_ok=True)

        self.bats = [
            Bat(self.parameters_df, self.output_dir)
            for _ in range(int(self.parameters_df["NUM_BATS"]))
        ]

        self.bats[0].allocentric_axis_y = Vector(-1, np.sqrt(3))

        obstacle_position = "random"
        obstacle_radius = self.parameters_df["OBSTACLE_RADIUS"]
        self.obstacles = [
            Obstacle(self.parameters_df, obstacle_position, obstacle_radius)
            for _ in range(int(self.parameters_df["OBSTACLE_COUNT"]))
        ]
        self.jammers = make_jammers(self.parameters_df)
        if self.parameters_df["ARENA_TYPE"] == "bounded":
            self.wall_objects = make_walls(self.parameters_df)
        elif self.parameters_df["ARENA_TYPE"] == "open":
            self.wall_objects = []
        else:
            raise ValueError("Unrecognised arena type")

        self.sound_objects = []  # Contains both DirectSound and EchoSound

        self.time_elapsed = 0.0
        self.history = []
        time_step_size = self.parameters_df["TIME_STEP"]
        # find the number of decimal places to set rounding equal to time step size
        self.store_history = store_history
        self.store_data_for_plotting = store_data_for_plotting
        self.rounding_based_on_time_step = len(str(time_step_size).split(".")[1])

        self.termination_criteria = termination_criteria

    def run(self):
        """Runs one instance of the simulation.
        After parsing the parameter file, it runs one instance of the simulation
        for those sets of parameters.
        """
        with open(self.dir_to_store + "bats_initial.pkl", "wb") as f:
            pickle.dump(self.bats, f)
        with open(self.dir_to_store + "obstacles_initial.pkl", "wb") as f:
            pickle.dump(self.obstacles, f)
        with open(self.dir_to_store + "jammers_initial.pkl", "wb") as f:
            pickle.dump(self.jammers, f)
        temporal_masking_fn_choice = self.parameters_df["TEMPORAL_MASKING_FN_TYPE"]

        if temporal_masking_fn_choice == "chopped":
            temporal_masking_file = read_temporal_masking_fn(
                "./src/bat-swarm-model/core/supporting_files/temporal_masking_fn_chopped.csv"
            )
        elif temporal_masking_fn_choice == "full":
            temporal_masking_file = read_temporal_masking_fn(
                "./src/bat-swarm-model/core/supporting_files/temporal_masking_fn.csv"
            )
        else:
            raise ValueError("unsupported temporal masking fn")

        # save initial snapshot of the simulation
        self.history.append(self.convert_necessary_information_into_dict())

        start_timing = datetime.now()
        # list_time_taken_for_each_loop = []
        # save_time_of_last_iter = start_timing
        if self.termination_criteria == "time":
            current_loop_timing = self.run_simulation_for_given_time(
                temporal_masking_file
            )
        elif self.termination_criteria == "distance":
            current_loop_timing = self.run_simulation_for_given_agent_distance(
                temporal_masking_file
            )
        else:
            raise ValueError("unsupported termination criteria")

        if self.store_history:
            handle_data_storage(
                self.time_elapsed,
                True,
                self.history,
                self.parameters_df,
                self.dir_to_store,
                self.store_data_for_plotting,
            )
            # self.handle_data_storage_for_plotting(self.time_elapsed, True)

        print(self.parameters_df["OUTPUT_DIR_FOR_SIMULATION"])
        print(f"total_time_taken_to_store_info: {current_loop_timing-start_timing}")
        # print(f"average_time_per_loop {np.mean(list_time_taken_for_each_loop)}")
        if self.store_history:
            print("DATA SAVED")

    def run_simulation_for_given_time(self, temporal_masking_file):
        time_array = np.round(
            np.arange(
                0, self.parameters_df["SIM_DURATION"], self.parameters_df["TIME_STEP"]
            ),
            self.rounding_based_on_time_step,
        )

        for time_step in tqdm(time_array[1:]):
            # print(time_step)
            # try:
            self.time_elapsed = time_step

            for sound in self.sound_objects:
                sound.update(self.time_elapsed)

            for bat in self.bats:
                bat.update(self.time_elapsed, self.sound_objects, temporal_masking_file)

            for jammer in self.jammers:
                jammer.update(self.time_elapsed, self.sound_objects)

            self.handle_reflections(self.time_elapsed)

            hearing_threshold = self.parameters_df["HEARING_THRESHOLD"]
            self.sound_objects = [
                s
                for s in self.sound_objects
                if s.active and s.current_spl > hearing_threshold
            ]

            self.history.append(self.convert_necessary_information_into_dict())
            # print(self.convert_necessary_information_into_dict())
            current_loop_time = datetime.now()
            # list_time_taken_for_each_loop.append(
            #     current_loop_time - save_time_of_last_iter
            # )

            # save_time_of_last_iter = current_loop_time

            if self.store_history:
                clear_history = handle_data_storage(
                    self.time_elapsed,
                    False,
                    self.history,
                    self.parameters_df,
                    self.dir_to_store,
                    self.store_data_for_plotting,
                )
                self.history = (
                    [] if clear_history else self.history
                )  # clear history after storing to save memory.

        # except BaseException:
        #     print("Unexpected error:", sys.exc_info()[0])
        #     print(
        #         f"error in file {self.parameters_df["OUTPUT_DIR_FOR_SIMULATION"]}"
        #     )
        #     break
        return current_loop_time

    def run_simulation_for_given_agent_distance(self, temporal_masking_file):
        distance_covered_by_agents = [bat.log_distance_moved for bat in self.bats]
        min_distance_covered = np.min(distance_covered_by_agents)
        time_step = 0
        t = tqdm(desc="while loop", total=self.parameters_df["DISTANCE_CAP"])
        while min_distance_covered < self.parameters_df["DISTANCE_CAP"]:
            min_distance_covered_in_last_loop = min_distance_covered
            time_step += self.parameters_df["TIME_STEP"]

            try:
                self.time_elapsed = time_step

                for sound in self.sound_objects:
                    sound.update(self.time_elapsed)

                for bat in self.bats:
                    bat.update(
                        self.time_elapsed, self.sound_objects, temporal_masking_file
                    )

                for jammer in self.jammers:
                    jammer.update(self.time_elapsed, self.sound_objects)

                self.handle_reflections(self.time_elapsed)

                hearing_threshold = self.parameters_df["HEARING_THRESHOLD"]
                self.sound_objects = [
                    s
                    for s in self.sound_objects
                    if s.active and s.current_spl > hearing_threshold
                ]

                self.history.append(self.convert_necessary_information_into_dict())
                # print(self.convert_necessary_information_into_dict())
                current_loop_time = datetime.now()
                # list_time_taken_for_each_loop.append(
                #     current_loop_time - save_time_of_last_iter
                # )

                # save_time_of_last_iter = current_loop_time

                if self.store_history:
                    clear_history = handle_data_storage(
                        self.time_elapsed,
                        False,
                        self.history,
                        self.parameters_df,
                        self.dir_to_store,
                        self.store_data_for_plotting,
                    )
                    self.history = (
                        [] if clear_history else self.history
                    )  # clear history after storing to save memory.

            except BaseException:
                print("Unexpected error:", sys.exc_info()[0])
                print(
                    f"error in file {self.parameters_df["OUTPUT_DIR_FOR_SIMULATION"]}"
                )
                break

            distance_covered_by_agents = [bat.log_distance_moved for bat in self.bats]
            min_distance_covered = np.min(distance_covered_by_agents)
            t.update(min_distance_covered - min_distance_covered_in_last_loop)

        return current_loop_time

    def convert_necessary_information_into_dict(self):
        """Handles information storage of simulation.
        Every iteration a subset of the simulation data
        needs to be stored, for later processing.

        Returns:
            dictionary : dictionary containing current simulation data
        """
        f = _as_np_float
        bat_directions = []
        for bat in self.bats:
            normalized_direction = bat.direction.normalize()
            bat_directions.append(
                (f(normalized_direction.x), f(normalized_direction.y))
            )
        bat_positions = [(f(bat.position.x), f(bat.position.y)) for bat in self.bats]
        # local polarization/density for every bat, computed once per frame
        local_polarization_list, local_density_list = (
            compute_local_polarization_per_bat(bat_directions, bat_positions, 2)
        )
        local_polarization_mean, local_density_mean = aggregate_local_polarization(
            local_polarization_list, local_density_list, "mean"
        )

        dictionary_w_information = {
            "time": np.round(self.time_elapsed, self.rounding_based_on_time_step),
            # "bat_id": [bat.id for bat in self.bats],
            "bat_ipi_counters": [len(bat.emit_times) for bat in self.bats],
            "bat_call_time": [bat.emit_times[-1] for bat in self.bats],
            "bat_positions": bat_positions,
            "bat_directions": bat_directions,
            "bat_distance_covered": [bat.log_distance_moved for bat in self.bats],
            "is_bat_rotating": [bat.is_bat_rotating for bat in self.bats],
            # "sound_objects": [
            #     self.serialize_sound(s)
            #     for s in self.sound_objects
            #     if s.active and s.current_spl > self.parameters_df["HEARING_THRESHOLD"]
            # ],
            "sound_objects_count": len(self.sound_objects),
            "jammer_positions": [
                (f(jammer.position.x), f(jammer.position.y)) for jammer in self.jammers
            ],
            "jammer_directions": [
                (f(jammer.direction.x), f(jammer.direction.y))
                for jammer in self.jammers
            ],
            "next_dir_angle": [
                bat.next_direction.angle_between(_X_AXIS) for bat in self.bats
            ],
            "current_dir_angle": [
                bat.direction.angle_between(_X_AXIS) for bat in self.bats
            ],
            "bat_response_vector": [
                (f(bat.next_direction.x), f(bat.next_direction.y)) for bat in self.bats
            ],
            "response_type": [bat.response_type for bat in self.bats],
            "bat_ipi_matrix_for_consistency": [
                bat.ipi_matrix_for_consistency for bat in self.bats
            ],
            "bat_sum_matrix": [bat.memory_window_sum_matrix for bat in self.bats],
            "bat_ipi_matrix_for_sensory_volume": [
                bat.ipi_matrix_for_sensory_volume for bat in self.bats
            ],
            "bat_intensity_matrix_for_consistency": [
                bat.sum_grids_in_memory_intensity for bat in self.bats
            ],
            # "polarization_score": compute_polarization(bat_directions),
            # "local_polarization_score": local_polarization_mean,
            # "local_polarization_score_list": [
            #     local_polarization_list[bat.id] for bat in self.bats
            # ],
            # "local_density_score": local_density_mean,
            # "local_density_score_list": [
            #     local_density_list[bat.id] for bat in self.bats
            # ],
            # "milling_score": compute_milling(bat_directions, bat_positions),
        }

        return dictionary_w_information

    def handle_reflections(self, current_time):
        """Generates reflections of the sound objects.
        Soud objects can reflect off of obstacles and bats
        to generate EchoSound s.

        Args:
            current_time (float): Time, in seconds, for which the simualtion has been running.
        """
        new_echoes = handle_reflections(
            current_time,
            self.sound_objects,
            self.wall_objects,
            self.obstacles,
            self.bats,
        )
        self.sound_objects.extend(new_echoes)

    def serialize_sound(self, sound):
        """Serializes sounds into dictionaries.
        This is done for easier storage.

        Args:
            sound (EchoSound): input sound object to be serialized

        Returns:
            dict: data inside the sound obejct is serialized into a dict.
        """
        data = {
            "origin": (sound.origin.x, sound.origin.y),
            "radius": sound.current_radius,
            "spl": sound.current_spl,
            "emitter_id": sound.emitter_id,
            "type": "direct" if isinstance(sound, DirectSound) else "echo",
            "status": sound.active,
        }

        return data
