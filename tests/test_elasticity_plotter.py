"""Test module of elasticity_plotter."""

# import unittest
# from sys import path as sys_path
# from typing import Callable
# from unittest.mock import patch

# sys_path.append("../src")
# from elasticity.utils.elasticity_plotter import ElasticityPlotter
# from elasticity.utils.mock_elasticity_generator import MockElasticityGenerator


# class TestElasticityPlotter(unittest.TestCase):
#     """Test case class for testing the generate_mock_elasticity_data function.

#     This class tests the functionality of the plots method
#     within the ElasticityPlotter class. It utilizes a mock function to ensure the method
#     is called correctly with the expected arguments.
#     """

#     @patch('src.elasticity_plotter.ElasticityPlotter.plot_price_and_units_for_uid_one_graph')
#     def test_plot_price_and_units_for_uid_one_graph(self, mock_plot_function: Callable) -> None:
#         """Test whether plot_price_and_units_for_uid_one_graph is called.

#         This method creates a mock data set and then calls the
#         plot_price_and_units_for_uid_one_graph method with a subset
#         of this data. It then asserts that the mock plot function
#         is called once with the expected arguments.
#         """
#         mock_elasticity_generator = MockElasticityGenerator()
#         mock_data = mock_elasticity_generator.generate_mock_elasticity_data()

#         uid = mock_data['uid'].iloc[0]
#         ElasticityPlotter().plot_price_and_units_for_uid_one_graph(mock_data, uid=uid)

#         mock_plot_function.assert_called_once_with(mock_data, uid=uid)

#     @patch('src.elasticity_plotter.ElasticityPlotter.plot_price_and_units_for_uid_2_graphs')
#     def test_plot_price_and_units_for_uid_2_graphs(self, mock_plot_function: Callable) -> None:
#         """Test whether plot_price_and_units_for_uid_2_graphs is called.

#         This method creates a mock data set and then calls the
#         plot_price_and_units_for_uid_2_graphs method with a subset
#         of this data. It then asserts that the mock plot function
#         is called once with the expected arguments.
#         """
#         mock_elasticity_generator = MockElasticityGenerator()
#         mock_data = mock_elasticity_generator.generate_mock_elasticity_data()

#         uid = mock_data['uid'].iloc[0]
#         ElasticityPlotter().plot_price_and_units_for_uid_2_graphs(mock_data, uid=uid)

#         mock_plot_function.assert_called_once_with(mock_data, uid=uid)

# if __name__ == '__main__':
#     unittest.main()
