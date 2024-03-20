"""Test module of preprocessing."""

import unittest

import numpy as np
import pandas as pd

from src.preprocessing import calculate_grouped_month_data, round_price_effect


class TestPreprocessing(unittest.TestCase):
    """Test case class for testing preprocessing function."""

    def test_round_price_effect(self) -> None:
        """Test round_price_effect."""
        self.assertEqual(round_price_effect(9.54), 9.5)
        self.assertEqual(round_price_effect(9.99), 9.95)
        self.assertEqual(round_price_effect(15.82), 15.5)
        self.assertEqual(round_price_effect(103.5), 103)
        self.assertEqual(round_price_effect(687), 685)

    def test_grouped_months(self) -> None:
        """Test grouped_months."""
        data = {
            'conversions_most_common_shelf_price': [100, np.nan, np.nan],
            'views_most_common_shelf_price': [np.nan, 200, np.nan],
            'price': [10, 20, 30],
            'total_units': [50, np.nan, 70]}
        df_test = pd.DataFrame(data)
        result = calculate_grouped_month_data(df_test)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['units'].tolist(), [50.0, 0.0, 70.0])

        expected_sources = ['conversions', 'views', 'price_recommendations']
        self.assertEqual(result['source'].tolist() , expected_sources)

if __name__ == '__main__':
    unittest.main()
