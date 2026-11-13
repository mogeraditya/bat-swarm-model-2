# implementation from jitesh jawar + guttal et al 2019 noise induced schooling in fishes.
# breif exaplination
# take the unit direction vectors of all the agents and sum them up.
# score is the length of the resultant vector
import numpy as np


def compute_polarization(bat_directions):
    """compute the polarization of a frame of bat directions.
    Magnitude of the sum of the normalized direction vector of all the agents.

    Args:
        bat_directions (list): List containing Vectors of bat directions

    Returns:
        int: polarization score, ranges from 0 to 1
    """
    normalized_vectors = [i / np.linalg.norm(i) for i in bat_directions]
    summed_vector = np.mean(normalized_vectors, axis=0)
    magnitude_of_summed_vector = np.linalg.norm(summed_vector)
    return magnitude_of_summed_vector


if __name__ == "__main__":
    x = [(0, 1), (1, 0)]
    y = np.mean(x, axis=0)
    z = np.dot(y, y)
    print(y, z)
