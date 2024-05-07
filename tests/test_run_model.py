"""Test module of model/run_model."""

from sys import path as sys_path

import numpy as np
import pandas as pd
import pytest

sys_path.append("../src")
from elasticity.model.run_model import run_experiment


@pytest.mark.usefixtures("sample_data")
def test_run_experiment(sample_data: pd.DataFrame) -> None:
    """Test run_experiment function with sample data."""
    # Call the function under test
    result = run_experiment(sample_data)

    # Assert that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Assert that the result DataFrame has the expected columns
    expected_columns = [
        "linear_mean_relative_error",
        "linear_mean_a",
        "linear_mean_b",
        "linear_mean_elasticity",
        "linear_mean_r2",
        "linear_a",
        "linear_b",
        "linear_pvalue",
        "linear_r2",
        "linear_elasticity",
        "linear_relative_absolute_error",
        "linear_elasticity_error_propagation",
        "linear_aic",
        "power_mean_relative_error",
        "power_mean_a",
        "power_mean_b",
        "power_mean_elasticity",
        "power_mean_r2",
        "power_a",
        "power_b",
        "power_pvalue",
        "power_r2",
        "power_elasticity",
        "power_relative_absolute_error",
        "power_elasticity_error_propagation",
        "power_aic",
        "exponential_mean_relative_error",
        "exponential_mean_a",
        "exponential_mean_b",
        "exponential_mean_elasticity",
        "exponential_mean_r2",
        "exponential_a",
        "exponential_b",
        "exponential_pvalue",
        "exponential_r2",
        "exponential_elasticity",
        "exponential_relative_absolute_error",
        "exponential_elasticity_error_propagation",
        "exponential_aic",
        "best_model",
        "best_model_a",
        "best_model_b",
        "best_model_r2",
        "best_mean_relative_error",
        "best_relative_absolute_error",
        "best_model_elasticity",
        "best_model_elasticity_error_propagation",
        "best_model_aic",
        "median_quantity",
        "median_price",
        "quality_test",
        "quality_test_high",
        "quality_test_medium",
        "details",
    ]
    print((result.columns))
    assert set(result.columns) == set(expected_columns)
    # test the results of the 3 models
    assert np.array_equal(round(result["linear_mean_relative_error"], 1), [0.0])
    assert np.array_equal(result["linear_elasticity"], [-1.0])
    assert np.array_equal(round(result["power_mean_relative_error"], 1), [77.0])
    assert np.array_equal(result["power_elasticity"], [-1.13])
    assert np.array_equal(round(result["exponential_mean_relative_error"], 1), [24.5])
    assert np.array_equal(result["exponential_elasticity"], [-1.31])
    # test that the best model is linear
    assert np.array_equal(result["best_model"], ["linear"])
    # test that the best model values are the same as the linear model
    assert np.array_equal(result["best_model_a"], result["linear_a"])
    assert np.array_equal(result["best_model_b"], result["linear_b"])
    assert np.array_equal(result["best_model_r2"], result["linear_r2"])
    assert np.array_equal(
        result["best_mean_relative_error"], result["linear_mean_relative_error"]
    )
    assert np.array_equal(result["best_model_elasticity"], result["linear_elasticity"])
    assert np.array_equal(
        result["best_model_elasticity_error_propagation"],
        result["linear_elasticity_error_propagation"],
    )
    # test that quality test is True
    assert np.array_equal(result["quality_test"], [True])
