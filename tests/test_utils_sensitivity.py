"""Test module of model/run_model."""

import numpy as np
import pandas as pd

from elasticity.data.utils_sensitivity import compute_corr, run_granger_test


def test_run_granger_test_success() -> None:
    """Granger test success."""
    # Generate sample data
    series_1 = pd.Series([1, 2, 3, 4, 5, 4, 7, 8, 9, 10])
    series_2 = pd.Series([2, 3, 4, 4, 4, 7, 8, 9, 10, 11])
    max_lag = 2

    # Run the Granger test
    result = run_granger_test(series_1, series_2, max_lag)

    # Check the result
    assert result is not None
    assert "p_values" in result
    assert "causality" in result
    assert len(result["p_values"]) == max_lag
    assert isinstance(result["causality"], bool)


def test_run_granger_test_insufficient_data() -> None:
    """Granger test insufficient."""
    # Generate sample data with insufficient length
    series_1 = pd.Series([1, 2])
    series_2 = pd.Series([2, 3])
    max_lag = 2

    # Run the Granger test
    result = run_granger_test(series_1, series_2, max_lag)

    # Check the result
    assert result is None


def test_run_granger_test_constant_series() -> None:
    """Granger test constant_series."""
    # Generate sample data with constant series
    series_1 = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    series_2 = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    max_lag = 2

    # Run the Granger test
    result = run_granger_test(series_1, series_2, max_lag)

    # Check the result
    assert result is None


def test_run_granger_test_non_numeric_data() -> None:
    """Granger test non_numeric_data."""
    # Generate sample data with non-numeric values
    series_1 = pd.Series(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    series_2 = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    max_lag = 2

    # Run the Granger test
    result = run_granger_test(series_1, series_2, max_lag)

    # Check the result
    assert result is None


def test_compute_corr_success() -> None:
    """compute_corr sucess."""
    # Generate sample data
    data = {"col1": [1, 2, 3, 4, 5], "col2": [2, 3, 4, 5, 6]}
    df_test = pd.DataFrame(data)

    # Compute correlation
    result = round(compute_corr(df_test, "col1", "col2"), 1)

    # Check the result
    assert result == 1.0


def test_compute_corr_missing_columns() -> None:
    """compute_corr missing columns."""
    # Generate sample data
    data = {"col1": [1, 2, 3, 4, 5]}
    df_test = pd.DataFrame(data)

    # Compute correlation
    result = compute_corr(df_test, "col1", "col2")

    # Check the result
    assert np.isnan(result)


def test_compute_corr_with_nan_values() -> None:
    """compute_corr nan values."""
    # Generate sample data with NaN values
    data = {"col1": [1, 2, np.nan, 4, 5], "col2": [2, 3, 4, 5, 6]}
    df_test = pd.DataFrame(data)

    # Compute correlation
    result = compute_corr(df_test, "col1", "col2")

    # Check the result
    assert result == 1.0


def test_compute_corr_insufficient_data() -> None:
    """compute_corr insufficient data."""
    # Generate sample data with insufficient length
    data = {"col1": [1], "col2": [2]}
    df_test = pd.DataFrame(data)

    # Compute correlation
    result = compute_corr(df_test, "col1", "col2")

    # Check the result
    assert result == 0


def test_compute_corr_zero_variance() -> None:
    """compute_corr zero variance."""
    # Generate sample data with zero variance in one column
    data = {"col1": [1, 1, 1, 1, 1], "col2": [2, 3, 4, 5, 6]}
    df_test = pd.DataFrame(data)

    # Compute correlation
    result = compute_corr(df_test, "col1", "col2")

    # Check the result
    assert result == 0


def test_compute_corr_with_infinite_values() -> None:
    """compute_corr infinite values."""
    # Generate sample data with infinite values
    data = {"col1": [1, 2, np.inf, 4, 5], "col2": [2, 3, 4, 5, 6]}
    df_test = pd.DataFrame(data)

    # Compute correlation
    result = compute_corr(df_test, "col1", "col2")

    # Check the result
    assert result == 1.0
