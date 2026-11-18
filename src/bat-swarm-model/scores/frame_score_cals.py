import math

import numpy as np
from scipy.spatial.distance import cdist


def normalize_rows(vectors):
    """Normalize row vectors, returning zero for zero-length vectors."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors),
        where=norms != 0,
    )


def compute_frame_scores(
    bat_directions,
    bat_positions,
    neighbourhood_distances,
):
    """
    Compute polarization, milling, and mean local metrics for one frame.

    The distance matrix and normalized directions are calculated only once.

    Returns:
        polarization: float
        milling: float
        local_polarizations: ndarray, shape (number_of_distances,)
        local_densities: ndarray, shape (number_of_distances,)
    """
    directions = np.asarray(bat_directions, dtype=float)
    positions = np.asarray(bat_positions, dtype=float)
    distances = np.asarray(neighbourhood_distances, dtype=float)

    if len(positions) == 0:
        empty = np.full(len(distances), np.nan)
        return np.nan, np.nan, empty, empty.copy()

    if np.any(distances <= 0):
        raise ValueError("Neighbourhood distances must be greater than zero")

    # Normalize only once for the entire frame.
    unit_directions = normalize_rows(directions)

    # Global polarization.
    polarization = np.linalg.norm(unit_directions.mean(axis=0))

    # Milling.
    centroid = positions.mean(axis=0)
    radial_directions = normalize_rows(positions - centroid)
    cross_products = np.cross(radial_directions, unit_directions)
    milling = np.linalg.norm(cross_products.mean(axis=0))

    # Squared distances avoid the unnecessary square root operation.
    squared_distances = cdist(positions, positions, metric="sqeuclidean")

    local_polarizations = np.empty(len(distances), dtype=float)
    local_densities = np.empty(len(distances), dtype=float)

    for distance_index, distance in enumerate(distances):
        # This preserves the original strict `< threshold` behavior.
        neighbourhood_mask = squared_distances < distance * distance

        # Self is included in each neighbourhood.
        neighbourhood_counts = neighbourhood_mask.sum(axis=1)

        # Sum all direction vectors in each bat's neighbourhood.
        summed_directions = neighbourhood_mask @ unit_directions

        per_bat_polarization = np.divide(
            np.linalg.norm(summed_directions, axis=1),
            neighbourhood_counts,
            out=np.zeros(len(positions), dtype=float),
            where=neighbourhood_counts > 0,
        )

        # Preserve original behavior: score is zero if there is no other bat.
        per_bat_polarization[neighbourhood_counts <= 1] = 0.0

        neighbourhood_area = math.pi * distance * distance
        per_bat_density = (neighbourhood_counts - 1) / neighbourhood_area

        local_polarizations[distance_index] = per_bat_polarization.mean()
        local_densities[distance_index] = per_bat_density.mean()

    return (
        polarization,
        milling,
        local_polarizations,
        local_densities,
    )
