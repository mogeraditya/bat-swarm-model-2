import os
import random
import sys
import uuid
from pathlib import Path
from typing import Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import optuna
import seaborn as sns
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)


from data_storage.combine_files_post_storage import (
    stitch_together_pickle_history_lists_and_store,
)
from plotting.visualizer import visualize
from scores.collision_scores import compute_collision_counts_and_length
from scores.local_polarization_score import compute_local_polarization
from scores.milling_score import compute_milling
from scores.run_all_score_calculations import (
    filter_bat_positions_from_history,
    given_history_list_generate_score_files,
    given_history_list_generate_score_files_for_sensitivity,
    reformat_history,
)
from simulation.class_simulation import Simulation
from supporting_files.utilities import load_parameters


@hydra.main(
    version_base=None, config_path="../core/config/", config_name="optimization_config"
)
def run_simulation_w_score_calcs(cfg: DictConfig) -> Tuple[float, float]:
    hydra_job_dir = HydraConfig.get().runtime.output_dir
    hydra_parent_dir = str(Path(hydra_job_dir).parent)
    job_uuid = uuid.uuid4()

    # fail safe for storage
    try:
        hydra_job_num = getattr(HydraConfig.get().job, "num", uuid.uuid4())
    except omegaconf.errors.MissingMandatoryValue:
        hydra_job_num = job_uuid

    OUTPUT_DIR = hydra_job_dir

    sim = Simulation(
        cfg,
        OUTPUT_DIR,
        store_history=True,
        store_data_for_plotting=True,
        termination_criteria="distance",
    )
    sim.run()

    history_list = stitch_together_pickle_history_lists_and_store(
        OUTPUT_DIR,
        hydra_parent_dir + "/data_file_storage/",
        f"job_number_{hydra_job_num}",
    )

    given_history_list_generate_score_files(
        history_list,
        hydra_job_dir,
        job_uuid,
        hydra_job_num,
        range_of_distance_for_local_score=np.arange(0.5, 4, 0.5),
    )
    distance_covered = history_list[-1]["bat_distance_covered"][0]
    parameters_df = load_parameters(OUTPUT_DIR + "/parameters_used.json")
    if distance_covered < parameters_df["DISTANCE_CAP"]:
        return np.inf, -np.inf

    bat_positions_list, bat_directions_list = filter_bat_positions_from_history(
        history_list
    )
    bat_positions_list = bat_positions_list[1000:]
    bat_directions_list = bat_directions_list[1000:]
    (
        wall_collision,
        bat_bat_collision,
        timeseries_wall_collision,
        timeseries_bat_bat_collision,
    ) = compute_collision_counts_and_length(bat_positions_list, parameters_df)

    pol_time_series = [
        compute_local_polarization(
            bat_directions, bat_positions_list[i], parameters_df, 2, "mean"
        )[0]
        for i, bat_directions in enumerate(bat_directions_list)
    ]

    mean_pol_of_group = np.mean(pol_time_series[1000:])

    # return total_collision_rate, max_pol_of_group
    return (bat_bat_collision, mean_pol_of_group)


if __name__ == "__main__":
    run_simulation_w_score_calcs()
