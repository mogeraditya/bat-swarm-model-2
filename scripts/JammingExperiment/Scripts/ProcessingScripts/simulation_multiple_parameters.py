import glob
import multiprocessing
import os
import random
import sys
import time

import numpy as np
from tqdm.rich import tqdm

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)
from data_storage.combine_files_post_storage import (  # merge_csvs,
    stitch_together_pickle_history_lists_and_store,
)
from scores.run_all_score_calculations import (
    given_history_list_generate_score_files_for_sensitivity,
)
from simulation.class_simulation import Simulation
from supporting_files.utilities import load_parameters, make_dir
from supporting_files.vectors import Vector


def run_one_instance_of_simulation(
    dir_of_one_param_file, data_storage_dir, initial_release_point
):
    """run one instance of the simulation

    Args:
        dir_of_one_param_file (str): directory of one param file
        simulation_id (int): used to track the iteration number of the sim

    """
    parameter_df = load_parameters(dir_of_one_param_file)
    random.seed(parameter_df["seed"])
    np.random.seed(parameter_df["seed"])
    # fail safe for storage
    job_num = parameter_df["PARAM_LABEL"]
    job_dir = parameter_df["OUTPUT_DIR_FOR_SIMULATION"]
    job_uuid = parameter_df["PARAM_UUID"]

    job_dir = (
        data_storage_dir
        + parameter_df["OUTPUT_DIR_FOR_SIMULATION"]
        # + f"iteration_number_{simulation_id}"
    )
    make_dir(job_dir)

    sim = Simulation(
        parameter_df,
        job_dir,
        store_history=True,
        store_data_for_plotting=True,
        use_hydra=False,
        termination_criteria="distance",
    )
    sim.bats[0].position = initial_release_point
    sim.bats[0].direction = Vector(1, 0)
    sim.run()

    history_list = stitch_together_pickle_history_lists_and_store(
        job_dir,
        job_dir,
        f"job_number_{job_num}",
    )

    given_history_list_generate_score_files_for_sensitivity(
        history_list,
        job_dir,
        job_uuid,
        job_num,
        range_of_distance_for_local_score=np.arange(0.5, 4, 0.5),
    )

    for filepath in os.listdir(job_dir):
        if "history" in filepath or ".json" in filepath:
            os.remove(job_dir + "/" + filepath)
        elif "_initial.pkl" in filepath:
            os.remove(job_dir + "/" + filepath)
        elif ".pkl.gz" in filepath:
            os.remove(job_dir + "/" + filepath)

    os.remove(dir_of_one_param_file)


def parallel_process_with_pool(
    param_dir,
    data_storage_dir,
    max_workers,
    initial_release_point,
):
    """run simulation multiple times for all parameter files

    Args:
        param_dir (str): directory containing all the parameter files
        n_runs (int): number of iteraitions per parameter file
        max_workers (int, optional): maximum number of cores that need to be used. Defaults to None.
    """
    # Find parameter files
    param_files = glob.glob(os.path.join(param_dir, "*.json"))
    param_files = [f for f in param_files if os.path.isfile(f)]
    print(param_files)

    if not param_files:
        print(f"No parameter files found in {param_dir}")
        return

    # Prepare tasks
    tasks = []
    for param_file in param_files:
        tasks.append(
            (
                param_file,
                data_storage_dir,
                initial_release_point,
            )
        )

    # Process with Pool
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    start_time = time.time()

    with multiprocessing.Pool(processes=max_workers) as pool:
        list(
            tqdm(
                pool.starmap_async(run_one_instance_of_simulation, tasks).get(),
                total=len(tasks),
                desc="Processing simulations",
                unit="task",
                ncols=100,
                colour="green",
            )
        )

    end_time = time.time()
    # merge_csvs(data_storage_dir, data_storage_dir + "summary_merged.csv")
    print(f"Pool processing completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # Directory containing your parameter files
    PARAM_DIR = "./scripts/JammingExperiment/Data/InputData/test/"

    N_RUNS = 1  # Number of iterations per parameter set
    DATA_STORAGE_DIR = r"./sensitivity_analysis_test_2/"  # Base output directory
    # MAX_WORKERS = 4  # Limit number of parallel processes

    # Run parallel processing
    print("Starting parallel processing...")

    chosen_start_location = Vector(5, 3.5)

    parallel_process_with_pool(
        param_dir=PARAM_DIR,
        data_storage_dir=DATA_STORAGE_DIR,
        max_workers=None,
        initial_release_point=chosen_start_location,
    )
