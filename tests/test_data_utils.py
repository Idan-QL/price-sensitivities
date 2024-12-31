"""Test module of model/run_model."""

import pandas as pd
import pytest

from elasticity.data.configurator import DataColumns
from elasticity.data.utils import (
    round_down_to_nearest_half,
    summarize_price_history,
)


def test_summarize_price_history() -> None:
    """Test summarize_price_history."""
    data = {
        "uid": ["A", "A", "A", "B", "B", "C"],
        "date": [
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
            "2023-01-01",
            "2023-02-01",
            "2023-01-01",
        ],
        "shelf_price": [10.0, 12.0, 11.0, 20.0, 22.0, 30.0],
    }
    input_df = pd.DataFrame(data)
    input_df["date"] = pd.to_datetime(input_df["date"])

    # Define data columns
    data_columns = DataColumns(
        uid="uid", date="date", shelf_price="shelf_price"
    )

    # Expected output
    expected_data = {
        "uid": ["A", "B", "C"],
        "last_date": ["2023-03-01", "2023-02-01", "2023-01-01"],
        "first_date": ["2023-01-01", "2023-01-01", "2023-01-01"],
        "min_price": [10.0, 20.0, 30.0],
        "max_price": [12.0, 22.0, 30.0],
        "last_price": [11.0, 22.0, 30.0],
    }

    expected_df = pd.DataFrame(expected_data)
    expected_df["last_date"], expected_df["first_date"] = pd.to_datetime(
        expected_df["last_date"]
    ), pd.to_datetime(expected_df["first_date"])

    result_df = summarize_price_history(input_df, data_columns)

    pd.testing.assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    "input_number, expected_output",
    [
        (3.7, 3.5),  # General rounding down
        (6.2, 6.0),  # Non-half decimal rounding
        (5.5, 5.5),  # Exact half
        (9.0, 9.0),  # Same float
        (0.0, 0.0),  # Zero
        (0.499999, 0.0),  # Just below 0.5
        (0.500001, 0.5),  # Just above 0.5
        (1e-10, 0.0),  # Very small fractional number
        (1e12 + 0.49, 1e12),  # Large number
        (1e12 + 0.51, 1e12 + 0.5),  # Large number
    ],
)
def test_round_down_to_nearest_half_positive_numbers(
    input_number: float, expected_output: float
) -> None:
    """Test round_down_to_nearest_half function with positive numbers."""
    assert round_down_to_nearest_half(input_number) == expected_output


@pytest.mark.parametrize(
    "input_number",
    [-0.5, -1.0, -1e-10, -1.49, -1.51, -1e6],
)
def test_round_down_to_nearest_half_negative_numbers(
    input_number: float,
) -> None:
    """Test round_down_to_nearest_half function with negative numbers."""
    with pytest.raises(ValueError):
        round_down_to_nearest_half(input_number)
