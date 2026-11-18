import json
import sys
import uuid
from itertools import product

import numpy as np

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)

from supporting_files.utilities import load_parameters, make_dir

simulation_parameters = load_parameters(
    r"./scripts/GroupExperiment/Data/InputData/common_parameters.json"
)
import random

dir_to_store = r"./scripts/GroupExperiment/Data/InputData/group_multirun/"
make_dir(dir_to_store)


# Only parameters with more than one value are included in the sweep.
NUM_BATS_VALUES = [5, 10, 30, 50]  # , 75, 100]

SEEDS = [
    132207,
    522173,
    710252,
    269432,
    152134,
    426499,
    892498,
    882110,
    915113,
    643227,
    # 901250,
    # 405197,
    # 719704,
    # 461562,
    # 314025,
    # 973687,
    # 422672,
    # 132467,
    # 294932,
    # 386342,
]

CONSISTENCY_TIME_DELAY_CAPS = [0.011, 0.014]
CONSISTENCY_WINDOWS = [1, 3, 5, 7]
LOUDNESS_TIME_DELAY_CAPS = [0.00]
arena_scaling_type = ["perimeter", "density", "fixed"]
CONTROLLER_TYPES = [
    "treatment",
    "positive_control",
    "negative_control",
    "null_control",
]
BAT_FRONTAL_RANGES = [60]


array_of_param_values = [
    NUM_BATS_VALUES,
    SEEDS,
    CONSISTENCY_TIME_DELAY_CAPS,
    CONSISTENCY_WINDOWS,
    LOUDNESS_TIME_DELAY_CAPS,
    CONTROLLER_TYPES,
    arena_scaling_type,
    BAT_FRONTAL_RANGES,
]

array_of_param_labels = [
    "NUM_BATS",
    "seed",
    "CONSISTENCY_TIME_DELAY_CAP",
    "MEMORY_WINDOW_FOR_CONSISTENCY",
    "LOUDNESS_TIME_DELAY_CAP",
    "control_type",
    "arena_size_scaling",
    "BAT_FRONTAL_RANGE",
]


list_of_combinations = list(product(*array_of_param_values))
random.shuffle(list_of_combinations)

print(f"Number of parameter combinations: {len(list_of_combinations)}")


output_dir_counter = 0

for item in list_of_combinations:
    simulation_parameters["OUTPUT_DIR_FOR_SIMULATION"] = (
        f"/group_multirun_analysis/paramset_{output_dir_counter}/"
    )

    for param_label, param_value in zip(array_of_param_labels, item):
        simulation_parameters[param_label] = param_value

    # Convert to int so that the value can be serialized by json.dump.
    simulation_parameters["NUMBER_OF_CONSISTENT_IPIS_FOR_MOVEMENT"] = int(
        np.ceil(simulation_parameters["MEMORY_WINDOW_FOR_CONSISTENCY"] / 2)
    )

    if simulation_parameters["arena_size_scaling"] == "fixed":
        simulation_parameters["ARENA_WIDTH"] = 9
        simulation_parameters["ARENA_LENGTH"] = 5
    elif simulation_parameters["arena_size_scaling"] == "perimeter":
        simulation_parameters["ARENA_WIDTH"] = (
            9 * (simulation_parameters["NUM_BATS"] / 30) + 4
        )
        simulation_parameters["ARENA_LENGTH"] = (
            5 * (simulation_parameters["NUM_BATS"] / 30) + 4
        )
    elif simulation_parameters["arena_size_scaling"] == "density":
        simulation_parameters["ARENA_WIDTH"] = (
            9 * np.sqrt(simulation_parameters["NUM_BATS"] / 30) + 4
        )
        simulation_parameters["ARENA_LENGTH"] = (
            5 * np.sqrt(simulation_parameters["NUM_BATS"] / 30) + 4
        )

    simulation_parameters["PARAM_LABEL"] = f"paramset_{output_dir_counter}"
    simulation_parameters["PARAM_UUID"] = str(uuid.uuid4())

    # Distribute the generated files across eight directories.
    mod_bin = output_dir_counter % 9
    output_subdir = dir_to_store + f"/group_multirun_{mod_bin}"
    make_dir(output_subdir)

    output_file = output_subdir + f"/paramset_number_{output_dir_counter}.json"

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(simulation_parameters, fp, indent=4)

    output_dir_counter += 1


print(f"Generated {output_dir_counter} parameter files.")
