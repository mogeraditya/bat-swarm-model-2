import subprocess

NUM_BATS_VALUES = [1]
SEEDS = [55, 89, 144, 233, 377]
CALL_RATES_FAST = [10, 20]
CALL_DIRECTIONALITYS = [4, 7, 10]
EMITTED_SPLS = [80, 100, 140]
HEARING_THRESHOLDS = [20, 40, 60, 80]
RADIAL_RESOLUTIONS = [0.002, 0.004, 0.006]
ANGULAR_RESOLUTIONS = [15, 30, 60]
BAT_FRONTAL_RANGES = [30, 60, 90]
CONSISTENCY_TIME_DELAY_CAPS = [0.008, 0.011, 0.014, 0.026]
CONSISTENCY_WINDOWS = [1, 3, 5, 7]


def main():
    subprocess.run(
        [
            "python",
            "./scripts/JammingExperiment/Scripts/ProcessingScripts/run_sweep.py",
            "-m",
            f"arena.NUM_BATS={','.join(map(str, NUM_BATS_VALUES))}",
            f"bat.CALL_RATE_FAST={','.join(map(str, CALL_RATES_FAST))}",
            f"bat.CALL_DIRECTIONALITY={','.join(map(str, CALL_DIRECTIONALITYS))}",
            f"bat.EMITTED_SPL={','.join(map(str, EMITTED_SPLS))}",
            f"bat.HEARING_THRESHOLD={','.join(map(str, HEARING_THRESHOLDS))}",
            f"bat.BAT_RADIAL_RESOLUTION={','.join(map(str, RADIAL_RESOLUTIONS))}",
            f"bat.BAT_ANGULAR_RESOLUTION={','.join(map(str, ANGULAR_RESOLUTIONS))}",
            f"bat.BAT_FRONTAL_RANGE={','.join(map(str, BAT_FRONTAL_RANGES))}",
            f"bat.CONSISTENCY_TIME_DELAY_CAP={','.join(map(str, CONSISTENCY_TIME_DELAY_CAPS))}",
            f"bat.MEMORY_WINDOW_FOR_CONSISTENCY={','.join(map(str, CONSISTENCY_WINDOWS))}",
            f"arena.seed={','.join(map(str, SEEDS))}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
