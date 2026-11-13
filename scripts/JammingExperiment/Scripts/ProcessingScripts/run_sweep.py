import os
import random
import sys
import uuid
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import seaborn as sns
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)

# from plotting.sim_plotter_single_bat_acoustic_field import visualize

from data_storage.combine_files_post_storage import (
    stitch_together_pickle_history_lists_and_store,
)

# from plotting.arena_module import ArenaModule
# from plotting.info_text_module import InfoTextModule
# from plotting.polarization_milling_module import PolarizationMillingModule
# from plotting.sensory_volume_module import SensoryVolumeModule
from plotting.visualizer import visualize
from scores.run_all_score_calculations import given_history_list_generate_score_files
from simulation.class_simulation import Simulation


@hydra.main(
    version_base=None,
    config_path="../../Data/InputData/config",
    config_name="slurm_optimize_config",
)
def run_simulation(cfg: DictConfig) -> None:
    random.seed(cfg.arena.seed)
    np.random.seed(cfg.arena.seed)

    hydra_job_dir = HydraConfig.get().runtime.output_dir
    hydra_parent_dir = str(Path(hydra_job_dir).parent)
    job_uuid = uuid.uuid4()

    # fail safe for storage
    try:
        hydra_job_num = getattr(HydraConfig.get().job, "num", uuid.uuid4())
    except omegaconf.errors.MissingMandatoryValue:
        hydra_job_num = job_uuid

    OUTPUT_DIR = hydra_job_dir

    print(OUTPUT_DIR)
    print(cfg)
    print(hydra_parent_dir)

    sim = Simulation(cfg, OUTPUT_DIR, store_history=True, store_data_for_plotting=True)
    sim.run()

    history_list = stitch_together_pickle_history_lists_and_store(
        OUTPUT_DIR,
        hydra_parent_dir + "/data_file_storage/",
        f"job_number_{hydra_job_num}",
    )

    df_summary, bat_bat_collision_counts, wall_collision_counts = (
        given_history_list_generate_score_files(
            history_list,
            hydra_job_dir,
            job_uuid,
            hydra_job_num,
            range_of_distance_for_local_score=np.arange(0.5, 4, 0.5),
        )
    )

    for filepath in os.listdir(hydra_job_dir):
        if ".pkl" in filepath:
            os.remove(hydra_job_dir + "/" + filepath)

    return bat_bat_collision_counts, wall_collision_counts


if __name__ == "__main__":
    run_simulation()
