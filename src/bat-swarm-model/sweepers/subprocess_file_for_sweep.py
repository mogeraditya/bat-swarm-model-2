import subprocess

# NUM_BATS_VALUES = [10]
# SEEDS = [1]
LOUDNESS_TIME_DELAY_CAPS = [0.030, 0.10]
# CONSISTENCY_TIME_DELAY_CAPS = [0.012]
# ANGLE_RANGE_FOR_ATTRACTIONS = [90]

# SPL_THRESHOLD_FOR_ATTRACTIONS = [85]
# SUSTAIN_SOUNDSS = [3]
N_JOBS = 12


def main():

    subprocess.run(
        [
            "python",
            "./src/bat-swarm-model/run_simulation/run_simulation_sweep.py",
            "-m",
            # f"arena.NUM_BATS={','.join(map(str, NUM_BATS_VALUES))}",
            f"bat.LOUDNESS_TIME_DELAY_CAP={','.join(map(str, LOUDNESS_TIME_DELAY_CAPS))}",
            # f"bat.CONSISTENCY_TIME_DELAY_CAP={','.join(map(str, CONSISTENCY_TIME_DELAY_CAPS))}",
            # f"bat.ANGLE_RANGE_FOR_ATTRACTION={','.join(map(str, ANGLE_RANGE_FOR_ATTRACTIONS))}",
            # f"bat.SPL_THRESHOLD_FOR_ATTRACTION={','.join(map(str, SPL_THRESHOLD_FOR_ATTRACTIONS))}",
            # f"bat.SUSTAIN_SOUNDS={','.join(map(str, SUSTAIN_SOUNDSS))}",
            # f"arena.seed={','.join(map(str, SEEDS))}",
            f"hydra.launcher.n_jobs={N_JOBS}",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
