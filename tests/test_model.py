import pandas as pd
import numpy as np
import pytest
from sys import path as sys_path
sys_path.append("../src")
from elasticity.model.model import estimate_coefficients
from tests.utils import round_tuple_values, sample_data

from typing import Tuple


def test_estimate_coefficients_linear(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with linear model.
    Expected result is a tuple of 5 floats.
    a, b, pvalue, r_squared, elasticity"""
    model_type = "linear"
    expected_result = (120.0, -2.0, 0.0, 1.0, -1.0)
    result = estimate_coefficients(sample_data, model_type)
    rounded_result = round_tuple_values(result)
    print("linear", rounded_result)
    assert rounded_result == expected_result

def test_estimate_coefficients_power(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with power model.
    Expected result is a tuple of 5 floats.
    a, b, pvalue, r_squared, elasticity"""
    model_type = "power"
    expected_result = (7.7, -1.1, 0.0, 0.8, -1.1)
    result = estimate_coefficients(sample_data, model_type)
    rounded_result = round_tuple_values(result)
    print("power", rounded_result)
    assert rounded_result == expected_result

def test_estimate_coefficients_exponential(sample_data: pd.DataFrame) -> None:
    """Test estimate_coefficients function with exponential model.
    Expected result is a tuple of 5 floats.
    a, b, pvalue, r_squared, elasticity"""
    model_type = "exponential"
    expected_result = (5.3, -0.0, 0.0, 0.9, -1.3)
    result = estimate_coefficients(sample_data, model_type)
    rounded_result = round_tuple_values(result)
    print("exponential", rounded_result)
    assert rounded_result == expected_result

