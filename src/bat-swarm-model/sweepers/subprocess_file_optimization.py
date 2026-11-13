import subprocess

import optuna


def main():
    subprocess.run(
        [
            "python",
            "./src/run_simulation/run_simulation_for_optima.py",
            "--multirun",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
