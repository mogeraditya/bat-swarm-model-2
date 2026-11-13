import sys

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
        "./scripts/JammingExperiment/Scripts/ProcessingScripts/",
    ]
)

from simulation_multiple_parameters import parallel_process_with_pool
from supporting_files.vectors import Vector

PARAM_DIR = "./scripts/JammingExperiment/Data/InputData/sensitivity_params/sensitivity_params_2/"
DATA_STORAGE_DIR = r"./sensitivity_analysis_2/"  # Base output directory
# MAX_WORKERS = 4  # Limit number of parallel processes

# Run parallel processing
print("Starting parallel processing...")

chosen_start_location = Vector(4.5, 2.5)

parallel_process_with_pool(
    param_dir=PARAM_DIR,
    data_storage_dir=DATA_STORAGE_DIR,
    max_workers=None,
    initial_release_point=chosen_start_location,
)
