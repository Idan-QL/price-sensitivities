"""Test module of preprocessing."""

from sys import path as sys_path

import numpy as np
import pandas as pd

sys_path.append("../src")
from elasticity.data.preprocessing import process_data, round_price_effect


def test_round_price_effect() -> None:
    """Test round_price_effect."""
    assert round_price_effect(9.54) == 9.5
    assert round_price_effect(9.99) == 9.95
    assert round_price_effect(15.82) == 15.5
    assert round_price_effect(103.5) == 103
    assert round_price_effect(687) == 685


def test_process_data() -> None:
    """Test process_data."""
    df_full = pd.DataFrame(
        {
            "conversions_most_common_shelf_price": [
                9.54,
                np.nan,
                15.82,
                np.nan,
                np.nan,
            ],
            "views_most_common_shelf_price": [np.nan, 9.99, np.nan, np.nan, np.nan],
            "price": [np.nan, np.nan, np.nan, np.nan, 687],
            "total_units": [10, np.nan, 30, np.nan, np.nan],
            "date": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
            ],
            "uid": [1, 2, 3, 4, 5],
        }
    )

    expected_result = pd.DataFrame(
        {
            "date": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
            ],
            "uid": [1, 2, 3, 4, 5],
            "round_price": [9.5, 9.95, 15.5, np.nan, 685],
            "units": [10.0, 0.0, 30.0, 0.0, 0.0],
            "price_merged": [9.54, 9.99, 15.82, np.nan, 687],
            "source": [
                "conversions",
                "views",
                "conversions",
                "price_recommendations",
                "price_recommendations",
            ],
        }
    )

    result = process_data(df_full)
    pd.testing.assert_frame_equal(result, expected_result)
