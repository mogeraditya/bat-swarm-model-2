import sys

import numpy as np
from scipy.stats import qmc

sys.path.extend(
    [
        "./src/bat-swarm-model/",
        "./src/bat-swarm-model/core/",
        "./src/bat-swarm-model/plot/",
    ]
)
from agents.class_obstacles import Obstacle
from simulation.class_simulation import cfg_to_flat_dict


def make_obstacles(cfg, radius, min_distance_between_clutter):
    parameters_df = cfg_to_flat_dict(cfg)
    engine = qmc.PoissonDisk(
        d=2,
        radius=min_distance_between_clutter,
        l_bounds=[min_distance_between_clutter, min_distance_between_clutter],
        u_bounds=[
            parameters_df["ARENA_WIDTH"] - min_distance_between_clutter,
            parameters_df["ARENA_LENGTH"] - min_distance_between_clutter,
        ],
    )
    sampled_points = engine.fill_space()

    store_clutter = [
        Obstacle(parameters_df, position, radius)
        for i, position in enumerate(sampled_points)
    ]
    return store_clutter, sampled_points
