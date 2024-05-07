"""Test module of model/model."""

from sys import path as sys_path

import pandas as pd

sys_path.append("../src")

from elasticity.model.model import estimate_coefficients
from tests.utils import round_tuple_values


# TODO Update test
def test_estimate_coefficients_linear(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with linear model.

    This function tests the estimate_coefficients function using a linear model.
    It expects the result to be a tuple of 5 floats: a, b, pvalue, r_squared, elasticity.

    Args:
        sample_data (pd.DataFrame): The sample data used for testing.

    Returns:
        None

    Raises:
        AssertionError: If the rounded result does not match the expected result.
    """
    model_type = "linear"
    expected_result = (120.0, -2.0, 0.0, 1.0, -1.0, 0.0)
    result = estimate_coefficients(sample_data, model_type)
    rounded_result = round_tuple_values(result)
    print("linear", rounded_result)
    assert rounded_result == expected_result


def test_estimate_coefficients_power(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with power model.

    This function tests the estimate_coefficients function using the power model.
    It expects the result to be a tuple of 5 floats: a, b, pvalue, r_squared, elasticity.

    Args:
        sample_data: A pandas DataFrame containing the sample data for testing.

    Returns:
        None

    Raises:
        AssertionError: If the rounded result does not match the expected result.
    """
    model_type = "power"
    expected_result = (7.7, -1.1, 0.0, 0.8, -1.1, 0.3)
    result = estimate_coefficients(sample_data, model_type)
    rounded_result = round_tuple_values(result)
    print("power", rounded_result)
    assert rounded_result == expected_result


def test_estimate_coefficients_exponential(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with exponential model.

    This function tests the estimate_coefficients function using an exponential model.
    It expects the result to be a tuple of 5 floats: a, b, pvalue, r_squared, elasticity.

    Args:
        sample_data (pd.DataFrame): The sample data used for testing.

    Returns:
        None

    Raises:
        AssertionError: If the rounded result does not match the expected result.
    """
    model_type = "exponential"
    expected_result = (5.3, -0.0, 0.0, 0.9, -1.3, 0.2)
    result = estimate_coefficients(sample_data, model_type)
    rounded_result = round_tuple_values(result)
    print("exponential", rounded_result)
    assert rounded_result == expected_result
