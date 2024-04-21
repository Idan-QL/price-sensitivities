"""Test module of mock_elasticity_generator.generate_mock_elasticity_data()."""

from datetime import datetime
import pandas as pd
from sys import path as sys_path

import pytest

sys_path.append("../src")
from elasticity.utils.mock_elasticity_generator import MockElasticityGenerator

@pytest.fixture
def mock_elasticity_generator():
    return MockElasticityGenerator()

def test_generate_nonlinear_elasticity(mock_elasticity_generator):
    """Test the generate_nonlinear_elasticity function to ensure it returns a DataFrame."""

    mock_data = mock_elasticity_generator.generate_nonlinear_elasticity()

    assert isinstance(mock_data, pd.DataFrame)
    # Check the length of the DataFrame
    assert len(mock_data) == 92 
    # Check if the DataFrame has the expected columns
    assert set(mock_data.columns) == {"date", "price", "uid", "units"} 

if __name__ == '__main__':
    pytest.main()