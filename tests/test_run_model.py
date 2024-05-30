"""Test module of model/run_model."""

from sys import path as sys_path

import numpy as np
import pandas as pd
import pytest

sys_path.append("../src")
from elasticity.model.run_model import run_experiment
from elasticity.utils.consts import OUTPUT_CS


@pytest.mark.usefixtures("sample_data")
def test_run_experiment(sample_data: pd.DataFrame) -> None:
    """Test run_experiment function with sample data."""
    # Call the function under test
    result = run_experiment(sample_data)

    # Assert that the result is a DataFrame
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == set(OUTPUT_CS)
    # test the results of the 3 models
    assert np.array_equal(result["linear_elasticity"], [-1.0])
    assert np.array_equal(result["power_elasticity"], [-1.13])
    assert np.array_equal(result["exponential_elasticity"], [-1.31])
    # test that the best model is linear
    assert np.array_equal(result["best_model"], ["linear"])
    # test that the best model values are the same as the linear model
    assert np.array_equal(result["best_a"], result["linear_a"])
    assert np.array_equal(result["best_b"], result["linear_b"])
    assert np.array_equal(result["best_r2"], result["linear_r2"])
    assert np.array_equal(result["best_elasticity"], result["linear_elasticity"])
    assert np.array_equal(
        result["best_elasticity_error_propagation"],
        result["linear_elasticity_error_propagation"],
    )
    # test that quality test is True
    assert np.array_equal(result["quality_test"], [True])
