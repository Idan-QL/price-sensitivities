"""Test module of utils."""

from typing import Tuple


def round_tuple_values(tup: Tuple[float], decimals: int = 1) -> Tuple[float]:
    """Round all values in a tuple to the specified number of decimals.

    Args:
        tup (Tuple[float]): The input tuple containing float values.
        decimals (int, optional): The number of decimal places to round to. Defaults to 1.

    Returns:
        Tuple[float]: The tuple with rounded values.

    """
    return tuple(round(value, decimals) for value in tup)
