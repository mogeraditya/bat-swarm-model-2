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
    r"./scripts/JammingExperiment/Data/InputData/common_parameters.json"
)
dir_to_store = r"./scripts/JammingExperiment/Data/InputData/sensitivity_params/"
make_dir(dir_to_store)


NUM_BATS_VALUES = [1]
SEEDS = [67, 69, 420, 80085, 1337, 1, 10, 100, 1000, 10000]
CALL_RATES_FAST = [10, 20]
CALL_DIRECTIONALITYS = [4, 7, 10]
# EMITTED_SPLS = [80, 100, 140]
HEARING_THRESHOLDS = [20, 40, 60]
RADIAL_RESOLUTIONS = [0.002, 0.004, 0.006]
ANGULAR_RESOLUTIONS = [15, 30, 60]
BAT_FRONTAL_RANGES = [30, 60, 90]
CONSISTENCY_TIME_DELAY_CAPS = [0.008, 0.011, 0.014]
CONSISTENCY_WINDOWS = [3, 5, 7, 10]

# number_of_consistent_ipis_for_movement = np.ceil(
#     np.array(memory_window_for_consistency) / 2
# )
# time_delay_for_repulsion = time_delay_for_direction_change

array_of_param_values = [
    NUM_BATS_VALUES,
    SEEDS,
    CALL_RATES_FAST,
    CALL_DIRECTIONALITYS,
    # EMITTED_SPLS,
    HEARING_THRESHOLDS,
    RADIAL_RESOLUTIONS,
    ANGULAR_RESOLUTIONS,
    BAT_FRONTAL_RANGES,
    CONSISTENCY_TIME_DELAY_CAPS,
    CONSISTENCY_WINDOWS,
]
array_of_param_labels = [
    "NUM_BATS",
    "seed",
    "CALL_RATE_FAST",
    "CALL_DIRECTIONALITY",
    # "EMITTED_SPL",
    "HEARING_THRESHOLD",
    "BAT_RADIAL_RESOLUTION",
    "BAT_ANGULAR_RESOLUTION",
    "BAT_FRONTAL_RANGE",
    "CONSISTENCY_TIME_DELAY_CAP",
    "MEMORY_WINDOW_FOR_CONSISTENCY",
]

list_of_combinations = list(product(*array_of_param_values))
print(len(list(product(*array_of_param_values))))

output_dir_counter = 0


for item in list_of_combinations:

    simulation_parameters["OUTPUT_DIR_FOR_SIMULATION"] = (
        f"/sensitivity_analysis/paramset_{output_dir_counter}/"
    )
    for i, param_label in enumerate(array_of_param_labels):
        simulation_parameters[param_label] = item[i]

    simulation_parameters["NUMBER_OF_CONSISTENT_IPIS_FOR_MOVEMENT"] = np.ceil(
        simulation_parameters["MEMORY_WINDOW_FOR_CONSISTENCY"] / 2
    )

    simulation_parameters["PARAM_LABEL"] = f"paramset_{output_dir_counter}"

    simulation_parameters["PARAM_UUID"] = str(uuid.uuid4())

    mod_bin = output_dir_counter % 8

    make_dir(dir_to_store + f"/sensitivity_params_{mod_bin}")
    with open(
        dir_to_store
        + f"/sensitivity_params_{mod_bin}/paramset_number_{output_dir_counter}.json",
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(simulation_parameters, fp)

    output_dir_counter += 1
