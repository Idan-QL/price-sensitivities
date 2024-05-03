"""Test module of utils."""

from typing import Tuple

import pandas as pd
import pytest


def round_tuple_values(tup: Tuple[float], decimals: int = 1) -> Tuple[float]:
    """Round all values in a tuple to the specified number of decimals."""
    return tuple(round(value, decimals) for value in tup)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture providing sample data for tests.

    those data are wollowing a linear demand curve with elasticity of -1 at median price of 30
    """
    return pd.DataFrame(
        {
            "price": [10, 20, 30, 40, 50],
            "quantity": [100, 80, 60, 40, 20],
            "days": [1, 2, 3, 4, 5],
        }
    )
