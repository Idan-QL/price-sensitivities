"""Test module of model/model."""

from sys import path as sys_path

import pandas as pd
import pytest

sys_path.append("../src")

from elasticity.model.model import estimate_coefficients


@pytest.mark.usefixtures("sample_data")
def test_estimate_coefficients_linear(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with linear model.

    This function tests the estimate_coefficients function using a linear model.
    It expects the result to be a tuple of 5 floats: a, b, pvalue, r2, elasticity.

    Args:
        sample_data (fixture Dataframe): The sample data used for testing.

    Returns:
        None

    Raises:
        AssertionError: If the rounded result does not match the expected result.
    """
    model_type = "linear"
    estimation_results = estimate_coefficients(sample_data, model_type)
    assert round(estimation_results.a, 1) == 120.0
    assert round(estimation_results.b, 1) == -2.0
    assert round(estimation_results.pvalue, 1) == 0.0
    assert round(estimation_results.r2, 1) == 1.0
    assert round(estimation_results.elasticity, 1) == -1.0
    assert round(estimation_results.elasticity_error_propagation, 1) == 0.0
    assert round(estimation_results.aic, 1) == -292.3
    assert round(estimation_results.relative_absolute_error, 1) == 0.0
    assert round(estimation_results.norm_rmse, 1) == 0.0


@pytest.mark.usefixtures("sample_data")
def test_estimate_coefficients_power(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with power model.

    This function tests the estimate_coefficients function using the power model.
    It expects the result to be a tuple of 5 floats: a, b, pvalue, r_squared, elasticity.

    Args:
        sample_data (fixture Dataframe): The sample data used for testing.

    Returns:
        None

    Raises:
        AssertionError: If the rounded result does not match the expected result.
    """
    model_type = "power"
    estimation_results = estimate_coefficients(sample_data, model_type)
    assert round(estimation_results.a, 1) == 7.7
    assert round(estimation_results.b, 1) == -1.1
    assert round(estimation_results.pvalue, 1) == 0.0
    assert round(estimation_results.r2, 1) == 0.8
    assert round(estimation_results.elasticity, 1) == -1.1
    assert round(estimation_results.elasticity_error_propagation, 1) == 0.3
    assert round(estimation_results.aic, 1) == 4.9
    assert round(estimation_results.relative_absolute_error, 1) == 21.8
    assert round(estimation_results.norm_rmse, 1) == 0.6


@pytest.mark.usefixtures("sample_data")
def test_estimate_coefficients_exponential(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with exponential model.

    This function tests the estimate_coefficients function using an exponential model.
    It expects the result to be a tuple of 5 floats: a, b, pvalue, r_squared, elasticity.

    Args:
        sample_data (fixture Dataframe): The sample data used for testing.

    Returns:
        None

    Raises:
        AssertionError: If the rounded result does not match the expected result.
    """
    model_type = "exponential"
    estimation_results = estimate_coefficients(sample_data, model_type)
    assert round(estimation_results.a, 1) == 5.3
    assert round(estimation_results.b, 1) == -0.0
    assert round(estimation_results.pvalue, 1) == 0.0
    assert round(estimation_results.r2, 1) == 0.9
    assert round(estimation_results.elasticity, 1) == -1.3
    assert round(estimation_results.elasticity_error_propagation, 1) == 0.2
    assert round(estimation_results.aic, 1) == -1.6
    assert round(estimation_results.relative_absolute_error, 1) == 12.1
    assert round(estimation_results.norm_rmse, 1) == 0.2
