"""Test moduling."""

import unittest
from sys import path as sys_path

sys_path.append("../src")
from elasticity.model.model import estimate_coefficients
from elasticity.utils.mock_elasticity_generator import MockElasticityGenerator


class TestElasticityPlotter(unittest.TestCase):
    def test_plot_price_and_units_for_uid_one_graph(self) -> None:

        test_elasticity = 5
        test_a = 1000
        mock_elasticity_generator = MockElasticityGenerator()
        mock_non_linear_elasticity = mock_elasticity_generator.generate_nonlinear_elasticity(
                start_date=mock_elasticity_generator.start_date,
                end_date=mock_elasticity_generator.end_date,
                elasticity=test_elasticity,
                a=test_a,
                price_range=mock_elasticity_generator.price_range,
                quantity_noise_std=0)

        a, b, pvalue, r_squared, elasticity = estimate_coefficients(data=mock_non_linear_elasticity,
                                                                    model_type='power',
                                                                    price_point=100,
                                                                    price_col= 'price',
                                                                    quantity_col='units')

        self.assertEqual(a, test_a)
        self.assertEqual(elasticity, test_elasticity)
        self.assertEqual(pvalue, 1)

if __name__ == '__main__':
    unittest.main()
