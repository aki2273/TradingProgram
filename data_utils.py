from logger import get_logger
log = get_logger()   

def parse_time_interval(interval: str) -> int:
    """
    Convert an interval string like '15m', '1h', '1d' … into minutes.
    Raises on a bad numeric part; logs an error and defaults to minutes
    when the unit is unknown.
    """
    unit = interval[-1]

    # ----- numeric part ----------------------------------------------------
    try:
        value = int(interval[:-1])
    except ValueError:
        log.error("Invalid interval \"%s\": numeric part missing or not an int.", interval)
        raise ValueError(f"Invalid time-interval string: {interval}") from None

    # ----- unit lookup -----------------------------------------------------
    mapping = {
        "m": 1,
        "h": 60,
        "d": 1440,
        "w": 10080,
        "M": 43800,          # ~30 days
    }

    if unit not in mapping:
        log.error(
            "Unknown time-unit '%s' in interval '%s'. "
            "Defaulting to minutes. (Supported: m, h, d, w, M)",
            unit, interval,
        )
        return value                    # treat as N minutes but warn the user

    return value * mapping[unit]

