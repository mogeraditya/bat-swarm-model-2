import bisect

import numpy as np
from supporting_files.utilities import str2bool


def convert_matrix_to_one_hot(m):
    return (m > 0).astype(m.dtype)


def convert_matrix_to_db(m):
    return np.where(m > 0, 20 * np.log10(m, where=m > 0), 0)


def given_parameters_df_return_grid_matrix_zeros(p):
    r = np.arange(
        p["CALL_DURATION"],
        p["CONSISTENCY_TIME_DELAY_CAP"] + p["BAT_RADIAL_RESOLUTION"],
        p["BAT_RADIAL_RESOLUTION"],
    )
    t = np.arange(-np.pi, np.pi, np.radians(p["BAT_ANGULAR_RESOLUTION"]))
    return np.zeros((len(r), len(t))), r, t


def convert_detected_sounds_into_grids(sounds, p, direction, axis):
    frame, dt = p["SPATIAL_REFERENCE_FRAME"], p["TIME_STEP"]
    decimals = len(str(dt).split(".")[1])
    grid, r, theta = given_parameters_df_return_grid_matrix_zeros(p)
    intensity = grid.copy()

    if not sounds:
        return grid, intensity, r, theta

    for s in sounds:
        delay = np.round(
            np.array(s["occurance_times"])[0] - s["bat_last_call_time"], decimals
        )
        if frame == "allocentric":
            angle = axis.angle_between(s["incident_direction"])
        elif frame == "egocentric":
            angle = direction.angle_between(s["incident_direction"])
        else:
            raise ValueError("not a supported spatial reference frame")

        i, j = bisect.bisect_left(r, delay), bisect.bisect_left(theta, angle) % len(
            theta
        )
        grid[i, j] += 1
        intensity[i, j] += 10 ** (s["received_spl"] / 20)

    grid = (
        convert_matrix_to_one_hot(grid)
        if str2bool(p["CONVERT_GRIDS_TO_ONE_HOT"])
        else grid / grid.max()
    )
    return grid, convert_matrix_to_db(intensity), r, theta


def given_matrix_shortlist_response_cells(m, threshold):
    cells = np.argwhere(m >= threshold)
    values = m[m >= threshold]

    if not len(cells):
        return np.array([[np.nan, np.nan]]), [[np.nan, np.nan]]

    np.random.shuffle(cells)
    cells = cells[cells[:, 0] == cells[:, 0].min()]
    return cells, [values[i] for i, x in enumerate(cells) if x[0] == cells[0, 0]]


def given_time_and_angle_return_direction(angle, p, direction, axis):
    resolution = np.radians(p["BAT_ANGULAR_RESOLUTION"])
    angle += -resolution / 2 + np.random.uniform(-resolution / 2, resolution / 2)

    if p["SPATIAL_REFERENCE_FRAME"] == "allocentric":
        angle += direction.angle_between(axis)

    return direction.rotate(angle + np.pi).normalize(), "repulsion"


def given_angle_find_cell_priorities(angle, columns):
    return np.argsort(np.abs(np.asarray(columns) - angle))


def find_indices_corresponding_to_hearing_range(p, theta, angle, hearing_range):
    resolution = np.radians(p["BAT_ANGULAR_RESOLUTION"])
    angles = np.arange(
        -hearing_range + angle, hearing_range + angle + resolution / 2, resolution
    )
    return [
        bisect.bisect_left(theta, x) % len(theta)
        for x in convert_angles_to_negpi_to_pi(angles)
    ]


def check_if_no_sound_in_front_of_bat(m, angle, n, p):
    active = (m >= n).sum(0)
    resolution = np.radians(p["BAT_ANGULAR_RESOLUTION"])
    theta = np.arange(-np.pi, np.pi, resolution)
    indices = find_indices_corresponding_to_hearing_range(
        p, theta, angle, np.radians(p["BAT_FRONTAL_RANGE"])
    )
    return not np.any(active[indices])


def make_sublist_based_on_hearing_range(activations, angle, p):
    resolution = np.radians(p["BAT_ANGULAR_RESOLUTION"])
    theta = np.arange(-np.pi, np.pi, resolution)
    indices = find_indices_corresponding_to_hearing_range(
        p, theta, angle, np.radians(p["HEARING_ANGLE_THRESHOLD"])
    )
    return np.asarray(activations)[indices], np.asarray(indices)


def convert_angles_to_negpi_to_pi(angles):
    angles = np.asarray(angles)
    return np.array(
        [a % (-np.sign(a) * np.pi) if abs(a) > np.pi else a for a in angles]
    )


def given_matrix_find_cell_to_respond_to_presence(
    m, intensity, n, p, direction, axis, previous
):
    angle = axis.angle_between(direction)

    if check_if_no_sound_in_front_of_bat(m, angle, n, p):
        return np.nan, np.nan

    # cells, values = given_matrix_shortlist_response_cells(m, n)
    cells = np.argwhere(m >= n)
    values = intensity[m >= n]
    # print(cells, values)
    if np.isnan(cells[0, 0]):
        return np.nan, np.nan

    if any(all(x in row for x in previous) for row in cells):
        return previous

    out = cells[np.argsort(values)[-1]]
    # print(out)
    return tuple(out)
