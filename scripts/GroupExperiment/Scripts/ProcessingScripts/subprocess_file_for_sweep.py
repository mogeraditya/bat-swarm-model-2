import subprocess

# import hydra

# Parameters to vary
NUM_BATS_VALUES = [5, 10, 30]
SEEDS = [
    1307,
    521,
    7152,
    2694,
    1514,
    4699,
    8998,
    8810,
    9513,
    642,
    9040,
    4097,
    7904,
    4662,
    3025,
    9767,
    426,
    1327,
    2992,
    3842,
]
CALL_RATES_FAST = [20]
CALL_DIRECTIONALITYS = [7]
HEARING_THRESHOLDS = [60]
RADIAL_RESOLUTIONS = [0.002]
ANGULAR_RESOLUTIONS = [15]
BAT_FRONTAL_RANGES = [30]
CONSISTENCY_TIME_DELAY_CAPS = [0.011, 0.014]
CONSISTENCY_WINDOWS = [5, 7]
LOUDNESS_TIME_DELAY_CAPS = [0.0, 0.030]
N_JOBS = 18
ARENA_SIZE_SCALINGS = ["fixed", "area", "perimeter"]


def main():
    subprocess.run(
        [
            "python",
            "./scripts/GroupExperiment/Scripts/ProcessingScripts/run_simulation_sweep.py",
            "-m",
            f"arena.NUM_BATS={','.join(map(str, NUM_BATS_VALUES))}",
            f"arena.seed={','.join(map(str, SEEDS))}",
            f"bat.CALL_RATE_FAST={','.join(map(str, CALL_RATES_FAST))}",
            f"bat.CALL_DIRECTIONALITY={','.join(map(str, CALL_DIRECTIONALITYS))}",
            f"bat.HEARING_THRESHOLD={','.join(map(str, HEARING_THRESHOLDS))}",
            f"bat.BAT_RADIAL_RESOLUTION={','.join(map(str, RADIAL_RESOLUTIONS))}",
            f"bat.BAT_ANGULAR_RESOLUTION={','.join(map(str, ANGULAR_RESOLUTIONS))}",
            f"bat.BAT_FRONTAL_RANGE={','.join(map(str, BAT_FRONTAL_RANGES))}",
            f"bat.CONSISTENCY_TIME_DELAY_CAP={','.join(map(str, CONSISTENCY_TIME_DELAY_CAPS))}",
            f"bat.MEMORY_WINDOW_FOR_CONSISTENCY={','.join(map(str, CONSISTENCY_WINDOWS))}",
            f"bat.LOUDNESS_TIME_DELAY_CAP={','.join(map(str, LOUDNESS_TIME_DELAY_CAPS))}",
            f"arena.arena_size_scaling={','.join(map(str, ARENA_SIZE_SCALINGS))}",
            f"hydra.launcher.n_jobs={N_JOBS}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
