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
from make_obstacles_based_on_poisson_disk_sample import make_obstacles

# from plotting.arena_module import ArenaModule
# from plotting.info_text_module import InfoTextModule
# from plotting.polarization_milling_module import PolarizationMillingModule
# from plotting.sensory_volume_module import SensoryVolumeModule
from plotting.visualizer import visualize
from scipy.spatial import ConvexHull
from scores.run_all_score_calculations import (
    given_history_list_generate_score_files_for_sensitivity,
)
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
    version_base=None, config_path="../../Data/InputData/config/", config_name="config"
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

    clutter_objects, clutter_positions = make_obstacles(
        cfg, radius=0.02, min_distance_between_clutter=cfg.arena.poisson_disk_radius
    )

    sim = Simulation(
        cfg,
        OUTPUT_DIR,
        store_history=True,
        store_data_for_plotting=True,
        termination_criteria="distance",
    )
    sim.obstacles.extend(clutter_objects)
    sim.run()

    history_list = stitch_together_pickle_history_lists_and_store(
        OUTPUT_DIR,
        hydra_parent_dir + "/data_file_storage/",
        f"job_number_{hydra_job_num}",
    )

    given_history_list_generate_score_files_for_sensitivity(
        history_list,
        hydra_job_dir,
        job_uuid,
        hydra_job_num,
        range_of_distance_for_local_score=np.arange(0.5, 4, 0.5),
    )

    # visualize(
    #     output_dir=OUTPUT_DIR,
    #     save_animation=hydra_parent_dir,
    #     unique_id=f"job_number_{hydra_job_num}",
    #     resolution=50,
    #     show_sounds=False,
    #     focal_bat=0,
    #     modules=[
    #         "arena",
    #         # "polarization_milling",
    #         "info_text",
    #         # "ipi_radial",
    #         # "sum_radial",
    #         "sensory_volume",
    #     ],
    # )

    sns.set_theme(style="whitegrid", context="talk")

    positions = np.array([i["bat_positions"][0] for i in history_list][::30])
    # print(positions)
    x, y = positions[:, 0], positions[:, 1]

    fig, ax = plt.subplots(figsize=(9, 5))

    s_bat = real_size_to_scatter_size(0.25, fig, 9)
    s_obstacle = real_size_to_scatter_size(0.04, fig, 9)
    hull = ConvexHull(positions)

    sns.scatterplot(x=x, y=y, color="black", alpha=0.2, ax=ax, s=s_bat)
    ax.scatter(
        clutter_positions[:, 0],
        clutter_positions[:, 1],
        label=f"obstacles, pds_radius={cfg.arena.poisson_disk_radius}",
        s=s_obstacle,
        marker="+",
    )
    ax.scatter(*positions[0], c="#55A868", s=120, label="Start", zorder=3)
    ax.scatter(*positions[-1], c="#C44E52", s=120, label="End", zorder=3)
    for simplex in hull.simplices:
        ax.plot(positions[simplex, 0], positions[simplex, 1], "r-", linewidth=2)
    ax.fill(
        positions[hull.vertices, 0],
        positions[hull.vertices, 1],
        alpha=0.2,
        color="lightgreen",
    )

    ax.set(xlim=(0, 9), ylim=(0, 5), aspect="equal")
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(hydra_parent_dir + f"/traj_{hydra_job_num}.png")
    plt.close()


if __name__ == "__main__":
    run_simulation()
