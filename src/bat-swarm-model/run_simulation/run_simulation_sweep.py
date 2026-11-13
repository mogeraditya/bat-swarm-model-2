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
from scores.run_all_score_calculations import (
    given_history_list_generate_score_files,
    given_history_list_generate_score_files_for_sensitivity,
)
from simulation.class_simulation import Simulation


@hydra.main(version_base=None, config_path="../core/config/", config_name="config")
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

    visualize(
        output_dir=OUTPUT_DIR,
        save_animation=hydra_parent_dir,
        unique_id=f"job_number_{hydra_job_num}",
        resolution=50,
        show_sounds=False,
        focal_bat=0,
        modules=[
            "arena",
            "polarization_milling",
            "info_text",
            "ipi_radial",
            "sum_radial",
            "intensity",
        ],
    )


if __name__ == "__main__":
    run_simulation()
