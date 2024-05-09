import math
import random
import string
from typing import Any


def clamp_w_tol(
    value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
    tolerance: float | None = 5e-3,
) -> float:
    """
    Clamp the value to [vmin, vmax] range with tolerance.
    """
    if tolerance:
        assert vmin - tolerance <= value <= vmax + tolerance, value
    return max(vmin, min(vmax, value))


def clamp(
    value: float,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> float:
    return clamp_w_tol(value, vmin, vmax, tolerance=None)


def is_dict_of_list(x: Any) -> bool:
    if isinstance(x, dict):
        return all(isinstance(v, list) for v in x.values())
    else:
        return False


def list_of_dict_to_dict_of_list(
    data: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    return {key: [i[key] for i in data] for key in data[0]}


def is_list_of_dict(x: Any) -> bool:
    if isinstance(x, list):
        return all(isinstance(d, dict) for d in x)
    else:
        return False


def dict_of_list_to_list_of_dict(
    dl: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    return [dict(zip(dl, t)) for t in zip(*dl.values())]


# https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
def argsort(x: list[int | float], reverse: bool = False) -> list[int]:
    assert isinstance(x, list) and isinstance(x[0], (int, float))
    return sorted(range(len(x)), key=x.__getitem__, reverse=reverse)


def is_equal_struct(x: Any, y: Any, key: str | None = None) -> bool:
    """
    Recursively check if two structures are equal.
    Note: key is optional but useful to track recursive call.
    """
    if isinstance(x, dict):
        assert set(x.keys()) == set(y.keys()), (x, y, key)
        assert all([is_equal_struct(x[key], y[key], key) for key in x]), (
            x,
            y,
            key,
        )
    elif isinstance(x, list):
        assert len(x) == len(y), (x, y)
        assert all([is_equal_struct(a, b, key) for a, b in zip(x, y)]), (
            x,
            y,
            key,
        )
    elif isinstance(x, str):
        assert x == y, (x, y, key)
    elif isinstance(x, float):
        assert math.isclose(x, y), (x, y, key)
    elif isinstance(x, int):
        assert math.isclose(x, y), (x, y, key)
    else:
        # other types should not be changed.
        assert id(x) == id(y), (x, y, key)

    return True


def is_float_list(x: Any) -> bool:
    if isinstance(x, list):
        return is_float_list(x[0])
    elif isinstance(x, float):
        return True
    else:
        return False


def mock_string(max_size: int = 10) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(random.randint(1, max_size)))
