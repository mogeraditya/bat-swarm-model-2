def calling_effort(bat_object):
    """counts the number of calls emited by an agent

    Args:
        bat_object (Bat): bat object

    Returns:
        int: number of calls emited by the bat
    """
    return len(bat_object.emit_times)
