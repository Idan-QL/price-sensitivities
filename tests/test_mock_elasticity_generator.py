"""Test module of mock_elasticity_generator.generate_mock_elasticity_data()."""

import unittest

import pandas as pd

from sys import path as sys_path
sys_path.append("../src")
from elasticity.utils.mock_elasticity_generator import MockElasticityGenerator


class TestMockElasticityGenerator(unittest.TestCase):
    """Test case class for testing the generate_mock_elasticity_data function."""

    def test_generate_mock_elasticity_data(self) -> None:
        """Test the generate_mock_elasticity_data function to ensure it returns a DataFrame."""
        mock_elasticity_generator = MockElasticityGenerator()
        mock_data = mock_elasticity_generator.generate_mock_elasticity_data()
        self.assertIsInstance(mock_data, pd.DataFrame)

        # Additional checks if needed
        # For example, you can check if the DataFrame has the expected columns, etc.
        # self.assertIn('column_name', mock_data.columns)

if __name__ == '__main__':
    unittest.main()
