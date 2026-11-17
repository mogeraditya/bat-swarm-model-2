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

dir_to_store = r"./scripts/GroupExperiment/Data/InputData/group_multirun/"
make_dir(dir_to_store)


# Only parameters with more than one value are included in the sweep.
NUM_BATS_VALUES = [5]  # , 10, 30, 50, 75, 100]

SEEDS = [
    13207,
    # 52173,
    # 71521,
    # 26942,
    # 15214,
    # 42699,
    # 89198,
    # 88210,
    # 95113,
    # 64322,
    # 90140,
    # 40197,
    # 71904,
    # 46162,
    # 31025,
    # 97367,
    # 42262,
    # 13247,
    # 29932,
    # 38342,
]

CONSISTENCY_TIME_DELAY_CAPS = [0.011, 0.014]
CONSISTENCY_WINDOWS = [3, 5, 7]
LOUDNESS_TIME_DELAY_CAPS = [0.0]

CONTROLLER_TYPES = [
    "treatment",
    "positive_control",
    "negative_control",
    "null_control",
]


array_of_param_values = [
    NUM_BATS_VALUES,
    SEEDS,
    CONSISTENCY_TIME_DELAY_CAPS,
    CONSISTENCY_WINDOWS,
    LOUDNESS_TIME_DELAY_CAPS,
    CONTROLLER_TYPES,
]

array_of_param_labels = [
    "NUM_BATS",
    "seed",
    "CONSISTENCY_TIME_DELAY_CAP",
    "MEMORY_WINDOW_FOR_CONSISTENCY",
    "LOUDNESS_TIME_DELAY_CAP",
    "control_type",
]


list_of_combinations = list(product(*array_of_param_values))

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

    simulation_parameters["PARAM_LABEL"] = f"paramset_{output_dir_counter}"
    simulation_parameters["PARAM_UUID"] = str(uuid.uuid4())

    # Distribute the generated files across eight directories.
    mod_bin = output_dir_counter % 8
    output_subdir = dir_to_store + f"/group_multirun_test"  # _{mod_bin}"
    make_dir(output_subdir)

    output_file = output_subdir + f"/paramset_number_{output_dir_counter}.json"

    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(simulation_parameters, fp, indent=4)

    output_dir_counter += 1


print(f"Generated {output_dir_counter} parameter files.")
