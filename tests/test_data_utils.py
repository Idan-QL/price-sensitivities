"""Test module of model/run_model."""

import random

import numpy as np
import pandas as pd
import pytest

from elasticity.data.configurator import DataColumns
from elasticity.data.utils import (
    outliers_iqr_filtered,
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
    data_columns = DataColumns(uid="uid", date="date", shelf_price="shelf_price")

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


def test_outliers_iqr_filtered_empty_list() -> None:
    """Test outliers_iqr_filtered function with an empty list."""
    ys = []
    result = outliers_iqr_filtered(ys)
    expected = []
    assert result == expected


def test_outliers_iqr_filtered_single_value() -> None:
    """Test outliers_iqr_filtered function with a single-item list."""
    ys = [5.0]
    result = outliers_iqr_filtered(ys)
    expected = [False]
    assert result == expected


def test_outliers_iqr_filtered_identical_values() -> None:
    """Test outliers_iqr_filtered function with a list that contains identical values."""
    n = 100
    ys = [50.0] * n
    result = outliers_iqr_filtered(ys)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_no_outliers() -> None:
    """Test outliers_iqr_filtered function with a list that doesn't contain any outliers."""
    random.seed(24)
    n = 100
    ys = [random.uniform(1, 10) for _ in range(n)]
    result = outliers_iqr_filtered(ys)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_with_outliers() -> None:
    """Test outliers_iqr_filtered function with a list that contains outliers."""
    random.seed(24)
    n = 100
    ys = [random.uniform(1, 10) for _ in range(n - 1)] + [100.0]
    result = outliers_iqr_filtered(ys)
    expected = [False] * (n - 1) + [True]
    assert result == expected


def test_outliers_iqr_filtered_all_below_filter_threshold() -> None:
    """Test outliers_iqr_filtered function with a list of values below filter threshold."""
    random.seed(24)
    n = 100
    ys = [random.uniform(0.0001, 0.0005) for _ in range(n)]
    result = outliers_iqr_filtered(ys, filter_threshold=0.001)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_all_below_filter_threshold_with_outliers() -> None:
    """Test outliers_iqr_filtered with values below filter threshold and potential outlier."""
    random.seed(24)
    n = 100
    ys = [random.uniform(0.0, 0.05) for _ in range(n - 1)] + [0.99]
    result = outliers_iqr_filtered(
        ys,
        filter_threshold=1,
        range_threshold=0.05,
        outlier_threshold=0.05,
    )
    expected = [False] * n
    assert result == expected


def test_outliers_iqr_filtered_all_above_outlier_threshold() -> None:
    """Test outliers_iqr_filtered function with a list of values above outlier threshold."""
    random.seed(24)
    n = 100
    ys = [random.uniform(11, 20) for _ in range(n)]
    result = outliers_iqr_filtered(ys, outlier_threshold=10)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_high_range() -> None:
    """Test outliers_iqr_filtered function with a list of values of range above range threshold."""
    ys = [10.0, 5.0, 50.0, 10.0, 20.0, 35.0, 40.0, 45.0, 55.0, 1.0]
    result = outliers_iqr_filtered(ys, range_threshold=15)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_mixed_types() -> None:
    """Test outliers_iqr_filtered function with a list of mixed data types."""
    ys = [5, 5.0, 7.5, 8, 9]
    result = outliers_iqr_filtered(ys)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_with_nan() -> None:
    """Test outliers_iqr_filtered function with a list containing nan."""
    ys = [1.5, 3.2, 7.7, 4.7, np.nan]
    result = outliers_iqr_filtered(ys)
    expected = [False] * len(ys)
    assert result == expected


def test_outliers_iqr_filtered_with_custom_parameters() -> None:
    """Test outliers_iqr_filtered function with custom parameters."""
    random.seed(24)
    n = 100
    ys = [100.0] + [random.uniform(1, 15) for _ in range(n - 1)]
    result = outliers_iqr_filtered(
        ys,
        filter_threshold=1,
        outlier_threshold=15,
        range_threshold=10,
        q=1.0,
        quartile=25,
    )
    expected = [True] + [False] * (n - 1)
    assert result == expected


def test_outliers_iqr_filtered_outside_bounds() -> None:
    """Test outliers_iqr_filtered function on a list with values slightly outside the bounds."""
    ys = [6.99, 8.999, 9.0, 9.999, 10.0, 10.999, 11.0, 13.01]
    result = outliers_iqr_filtered(ys, range_threshold=5, outlier_threshold=3, quartile=20, q=1)
    expected = [True, False, False, False, False, False, False, True]
    assert result == expected


def test_outliers_iqr_filtered_inside_bounds() -> None:
    """Test outliers_iqr_filtered function on a list with values near but within bounds."""
    ys = [7.01, 8.999, 9.0, 9.999, 10.0, 10.999, 11.0, 12.99]
    result = outliers_iqr_filtered(ys, range_threshold=5, outlier_threshold=3, quartile=20, q=1)
    expected = [False, False, False, False, False, False, False, False]
    assert result == expected
