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
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver(
    "div", lambda a: int(np.ceil(float(a) / 2)), replace=True
)
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
from scores.run_all_score_calculations import given_history_list_generate_score_files
from simulation.class_simulation import Simulation


def real_size_to_scatter_size(real_size_m, fig, arena_width_m):
    """
    Convert real-world size (in meters) to matplotlib scatter s parameter.
    Assumes the figure width matches the arena width.
    """
    points_per_meter = (fig.get_size_inches()[0] / arena_width_m) * 72
    radius_points = real_size_m * 0.5 * points_per_meter  # radius in points
    return 3.14159 * (radius_points**2)  # area in points²


@hydra.main(
    version_base=None,
    config_path="../../Data/InputData/config/",
    config_name="config",
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

    if cfg.arena.arena_size_scaling == "fixed":
        cfg.arena.ARENA_WIDTH = 9
        cfg.arena.ARENA_LENGTH = 5
    elif cfg.arena.arena_size_scaling == "perimeter":
        cfg.arena.ARENA_WIDTH = 9 * (cfg.arena.NUM_BATS / 30) + 2
        cfg.arena.ARENA_LENGTH = 5 * (cfg.arena.NUM_BATS / 30) + 2
    elif cfg.arena.arena_size_scaling == "area":
        cfg.arena.ARENA_WIDTH = 9 * np.sqrt(cfg.arena.NUM_BATS / 30) + 2
        cfg.arena.ARENA_LENGTH = 5 * np.sqrt(cfg.arena.NUM_BATS / 30) + 2
    else:
        raise ValueError("wrong arena size")

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


if __name__ == "__main__":
    run_simulation()
