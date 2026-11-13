import subprocess

# import hydra

# Parameters to vary
NUM_BATS_VALUES = [1]
SEEDS = [666]
CALL_RATES_FAST = [20]
CALL_DIRECTIONALITYS = [7]
HEARING_THRESHOLDS = [60]
RADIAL_RESOLUTIONS = [0.002, 0.004]
ANGULAR_RESOLUTIONS = [15]
BAT_FRONTAL_RANGES = [30]
CONSISTENCY_TIME_DELAY_CAPS = [0.008, 0.011, 0.014]
CONSISTENCY_WINDOWS = [3, 5, 7]
JAMMER_RESOLUTIONS = [1000, 0.8, 0.5, 0.23]
N_JOBS = 15


def main():
    subprocess.run(
        [
            "python",
            "./scripts/JammingExperiment/Scripts/ProcessingScripts/test_best_params/run_test_for_best_params.py",
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
            f"arena.JAMMER_RESOLUTION={','.join(map(str, JAMMER_RESOLUTIONS))}",
            f"hydra.launcher.n_jobs={N_JOBS}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
