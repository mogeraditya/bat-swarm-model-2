import numpy as np
from agents.class_sounds import DirectSound
from supporting_files.utilities import creation_time_calculation

# candidate search margin; the final in/out decision is always the exact
# scalar ``contains_point`` so this only has to exceed float noise (~1e-15).
_CANDIDATE_MARGIN = 1e-9


class _WallGeometry:
    """Static wall data, built once per wall list (walls never move)."""

    def __init__(self, wall_objects_list):
        self.walls = wall_objects_list
        self.count = len(wall_objects_list)
        self.x = np.array([float(w.position.x) for w in wall_objects_list])
        self.y = np.array([float(w.position.y) for w in wall_objects_list])
        self.keys = [f"wall_obstacle_{w.id}" for w in wall_objects_list]
        # Jammers call from fixed positions in fixed directions, so the same
        # (sound origin, direction, level, wall) reflection geometry recurs on
        # every call. Everything cached is a pure function of the key.
        self.sorted_distance_cache = {}
        self.wall_echo_cache = {}


_CACHE_SIZE_LIMIT = 200_000


def _bounded_put(cache, key, value):
    if len(cache) >= _CACHE_SIZE_LIMIT:
        cache.clear()
    cache[key] = value


_wall_geometry_cache = []


def _get_wall_geometry(wall_objects_list):
    for geometry in _wall_geometry_cache:
        if (
            geometry.walls is wall_objects_list
            and geometry.count == len(wall_objects_list)
        ):
            return geometry
    geometry = _WallGeometry(wall_objects_list)
    _wall_geometry_cache[:] = [geometry]  # one simulation at a time
    return geometry


def _sorted_wall_distances(sound, geometry):
    """Distances from a (fixed) sound origin to every (fixed) wall, sorted.
    Computed once per sound instead of once per sound per time step."""
    cached = getattr(sound, "_wall_distance_cache", None)
    if cached is None or cached[0] is not geometry:
        origin_key = (sound.origin.x, sound.origin.y)
        sorted_and_order = geometry.sorted_distance_cache.get(origin_key)
        if sorted_and_order is None:
            dx = np.round(sound.origin.x - geometry.x, 5)
            dy = np.round(sound.origin.y - geometry.y, 5)
            distances = np.sqrt(dx * dx + dy * dy)
            order = np.argsort(distances, kind="stable")
            sorted_and_order = (distances[order], order)
            _bounded_put(geometry.sorted_distance_cache, origin_key, sorted_and_order)
        cached = (geometry, *sorted_and_order)
        sound._wall_distance_cache = cached
    return cached[1], cached[2]


def _wall_echo_geometry(sound, geometry, wall_index, wall, wall_key):
    """(normal, reflection point, distance to it, echo spl or None) for a wall,
    computed with the original Vector operations and memoised."""
    key = (
        sound.origin.x,
        sound.origin.y,
        sound.direction_vector.x,
        sound.direction_vector.y,
        sound.initial_spl,
        wall_index,
    )
    cached = geometry.wall_echo_cache.get(key)
    if cached is None:
        normal = wall.get_reflection_normal(sound.origin)
        reflection_point = wall.position + normal * wall.radius
        # == creation_time_calculation's distance
        distance = (sound.origin - reflection_point).magnitude()
        reflected_spl = sound.reflected_spl_at(reflection_point, wall_key)
        cached = (normal, reflection_point, distance, reflected_spl)
        _bounded_put(geometry.wall_echo_cache, key, cached)
    return cached


def _walls_possibly_in_shell(sound, geometry):
    """Indices (in wall-list order) of walls that may lie in the sound shell.
    A superset: every index is re-checked exactly with ``contains_point``."""
    sorted_distances, order = _sorted_wall_distances(sound, geometry)
    radius = sound.current_radius
    inner_radius = max(0, radius - sound.shell_width)
    start = np.searchsorted(sorted_distances, inner_radius - _CANDIDATE_MARGIN, "left")
    stop = np.searchsorted(sorted_distances, radius + _CANDIDATE_MARGIN, "right")
    if start >= stop:
        return ()
    return sorted(order[start:stop].tolist())


def handle_reflections(current_time, sound_objects, wall_objects_list, obstacles, bats):
    """Generates reflections of the sound objects.
    Soud objects can reflect off of obstacles and bats
    to generate EchoSound s.

    Args:
        current_time (float): Time, in seconds, for which the simualtion has been running.
    """
    new_echoes = []

    wall_geometry = _get_wall_geometry(wall_objects_list)
    wall_keys = wall_geometry.keys
    obstacle_keys = [f"obstacle_{obstacle.id}" for obstacle in obstacles]
    bat_keys = [f"bat_{bat.id}" for bat in bats]

    for sound in sound_objects:
        if not sound.active or not isinstance(sound, DirectSound):
            continue

        reflection_point = None
        normal = None
        obstacle_id = None

        reflection_point_arr, normal_arr, obstacle_id_arr = [], [], []
        precomputed_arr = []  # (distance, echo spl) for walls, else None

        # check walls
        sound_wall_id = sound.wall_id
        reflected_obstacles = sound.reflected_obstacles
        for wall_index in _walls_possibly_in_shell(sound, wall_geometry):
            wall_objects = wall_objects_list[wall_index]
            wall_key = wall_keys[wall_index]
            # check only for jammer call emissions.
            # (cheap checks first; all three conditions are side-effect free)
            if (
                sound_wall_id != wall_objects.wall_id
                and wall_key not in reflected_obstacles
                and sound.contains_point(wall_objects.position)
            ):
                normal, reflection_point, distance, reflected_spl = (
                    _wall_echo_geometry(
                        sound, wall_geometry, wall_index, wall_objects, wall_key
                    )
                )
                obstacle_id = wall_key

                normal_arr.append(normal)
                reflection_point_arr.append(reflection_point)
                obstacle_id_arr.append(obstacle_id)
                precomputed_arr.append((distance, reflected_spl))
        # check obstacles
        for obstacle, obstacle_key in zip(obstacles, obstacle_keys):
            if obstacle_key not in reflected_obstacles and sound.contains_point(
                obstacle.position
            ):
                normal = obstacle.get_reflection_normal(sound.origin)
                reflection_point = obstacle.position + normal * obstacle.radius
                obstacle_id = obstacle_key

                normal_arr.append(normal)
                reflection_point_arr.append(reflection_point)
                obstacle_id_arr.append(obstacle_id)
                precomputed_arr.append(None)

        # check other bats
        for bat, bat_key in zip(bats, bat_keys):
            if (
                sound.emitter_id != bat.id
                and bat.is_bat_reflective_to_sound
                and bat_key not in reflected_obstacles
                and sound.contains_point(bat.position)
            ):
                normal = (sound.origin - bat.position).normalize()
                reflection_point = bat.position + normal * bat.radius
                obstacle_id = bat_key

                normal_arr.append(normal)
                reflection_point_arr.append(reflection_point)
                obstacle_id_arr.append(obstacle_id)
                precomputed_arr.append(None)

        for i, reflection_point in enumerate(reflection_point_arr):
            normal = normal_arr[i]
            obstacle_id = obstacle_id_arr[i]
            if obstacle_id not in sound.reflected_obstacles:
                precomputed = precomputed_arr[i]
                if precomputed is None:
                    time_of_creation = creation_time_calculation(
                        sound, reflection_point
                    )
                    echo = sound.create_echo(
                        reflection_point, time_of_creation, normal, obstacle_id
                    )
                else:
                    distance, reflected_spl = precomputed
                    # same arithmetic as creation_time_calculation
                    time_of_creation = distance / sound.speed + sound.creation_time
                    echo = sound.make_echo(
                        reflection_point,
                        time_of_creation,
                        normal,
                        obstacle_id,
                        reflected_spl,
                    )

                if echo:
                    # mark this obstacle as reflected for the original sound
                    echo.update(current_time)
                    sound.reflected_obstacles.add(obstacle_id)
                    # copy reflected obstacles to the echo
                    echo.reflected_obstacles.update(sound.reflected_obstacles)
                    new_echoes.append(echo)

    return new_echoes
