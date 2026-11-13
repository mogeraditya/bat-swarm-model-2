import numpy as np
import scipy as scp


def compute_time_spent_rotating(bat_history, parameters_df):
    """Calculate collisions given position history of bats

    Args:

        parameters_df (dict): parameter file of the simulation run

    Returns:
    """
    counter = [0 for i in bat_history[0]["is_bat_rotating"]]
    for frame in bat_history:
        for i, bat in enumerate(frame["is_bat_rotating"]):
            if bat:
                counter[i] += 1

    time_spent_rotating = [i * parameters_df["TIME_STEP"] for i in counter]
    return time_spent_rotating
