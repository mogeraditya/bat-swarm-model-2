import bisect

import numpy as np


def convex_hull_area(points):
    # Step 1: Compute convex hull (using Graham scan or similar)
    # Here we use a simple implementation for clarity (only for 2D)
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Sort points lexicographically
    points = sorted(points)

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenate to get hull (remove last point of each half to avoid duplication)
    hull = lower[:-1] + upper[:-1]

    # Step 2: Use shoelace formula to compute area
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    area = abs(area) / 2.0
    return area


def space_occupied_score(parameters_df, bat_positions, bat_id=0):
    """Score that handles space occupied by the agent in an arena
    Partition the space into coarse grid cells (0.5 m x 0.5 m) and count
    how many the agent walked over.
    Args:
        parameters_df (dict): parameter file of the simulation
        bat_positions (list of Vectors): position of bat at every timestep

    Returns:
        int: number of unique grid cells agent was in
    """

    points_of_bat_positions = []
    for bat_position_frame in bat_positions:
        points_of_bat_positions.append(
            [bat_position_frame[bat_id][0], bat_position_frame[bat_id][1]]
        )

    # hull = ConvexHull(points_of_bat_positions)
    area = convex_hull_area(points_of_bat_positions)
    print("Convex hull area:", area)
    return area
