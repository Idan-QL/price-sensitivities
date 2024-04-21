"""This module contains a class MockElasticityGenerator.

It Generates mock data for linear and non-linear demand based on price elasticity.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame


class MockElasticityGenerator:
    """Generates mock data for linear and non-linear demand based on price elasticity."""

    def __init__(
        self,
        start_date: datetime = datetime(2024, 10, 1),
        end_date: datetime = datetime(2024, 12, 31),
        price_range: tuple[int, int] = (100, 200),
        a_linear: float = 999,
        a_non_linear: float = 500,
        a_power_non_linear: float = 100,
        quantity_noise_std: float = 0,
        elasticity: float = -3,
        date_col: str = "date",
        price_col: str = "price",
        demand_col: str = "units",
        uid_col: str = "uid",
    ) -> None:

        """Initializes the MockElasticityGenerator with default parameters.

        Parameters:
        - elasticity: elasticities to generate date for.
        - start_date (datetime): Start date for mock data generation.
        - end_date (datetime): End date for mock data generation.
        - price_range (tuple): Tuple specifying the range of prices.
        - a_linear (float): Parameter 'a' for linear elasticity generation.
        - a_non_linear (float): Parameter 'a' for non-linear elasticity generation.
        - a_power_non_linear (float): Parameter 'a' for non-linear elasticity generation.
        - date_col (str): Name of the date column in the generated DataFrame.
        - price_col (str): Name of the price column in the generated DataFrame.
        - demand_col (str): Name of the demand column in the generated DataFrame.
        - uid_col (str): Name of the uid column in the generated DataFrame.
        """
        self.start_date = start_date
        self.end_date = end_date
        self.price_range = price_range
        self.a_linear = a_linear
        self.a_non_linear = a_non_linear
        self.a_power_non_linear = a_power_non_linear
        self.quantity_noise_std = quantity_noise_std
        self.date_col = date_col
        self.price_col = price_col
        self.demand_col = demand_col
        self.uid_col = uid_col
        self.elasticity = elasticity

    def generate_nonlinear_elasticity(self) -> pd.DataFrame:
        """Generates demand data based on a non-linear price elasticity model.

        Parameters:
        - start_date (datetime): Start date for demand generation.
        - end_date (datetime): End date for demand generation.
        - elasticity (float): Elasticity parameter for demand generation.
        - a (float): Parameter 'a' for demand generation.
        - price_range (tuple): Tuple specifying the range of prices.
        - quantity_noise_std (float): Standard deviation for quantity noise.

        Returns:
        - DataFrame: DataFrame containing demand data.
        """
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        min_price, max_price = self.price_range
        prices = np.random.uniform(min_price, max_price, len(date_range))
        prices = np.round(prices, 2)
        base_quantity = self.a_non_linear * np.power(prices, self.elasticity)
        quantity_noise = np.random.normal(
            scale=self.quantity_noise_std, size=len(date_range)
        )
        quantity = base_quantity + quantity_noise
        df_nonlinear = pd.DataFrame(
            {
                self.date_col: date_range,
                self.price_col: prices,
                self.demand_col: quantity,
                self.uid_col: "uid_nonlinear_" + str(self.elasticity),
            }
        )
        df_nonlinear["units"] = df_nonlinear["units"].round(3)
        return df_nonlinear

    def generate_linear_elasticity(self) -> pd.DataFrame:
        """Generates demand data based on a linear price elasticity model.

        TODO: Delete or change as elasticity not constant
        Parameters:
        - start_date (datetime): Start date for demand generation.
        - end_date (datetime): End date for demand generation.
        - elasticity (float): Elasticity parameter for demand generation.
        - a (float): Parameter 'a' for demand generation.
        - price_range (tuple): Tuple specifying the range of prices.
        - quantity_noise_std (float): Standard deviation for quantity noise.

        Returns:
        - DataFrame: DataFrame containing demand data.
        """
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq="D")
        start_price, end_price = self.price_range
        prices = np.linspace(start_price, end_price, len(date_range))
        base_quantity = self.a_linear + (self.elasticity) * prices
        quantity_noise = np.random.normal(
            scale=self.quantity_noise_std, size=len(date_range)
        )
        quantity = base_quantity + quantity_noise
        df_linear = pd.DataFrame(
            {
                self.date_col: date_range,
                self.price_col: prices,
                self.demand_col: quantity,
                self.uid_col: "uid_linear_" + str(self.elasticity),
            }
        )
        df_linear["units"] = df_linear["units"].round(3).clip(lower=0)
        return df_linear

    def generate_mock_elasticity_data(self) -> pd.DataFrame:
        """Generates a DataFrame with mock data for linear and non-linear demand.

        Returns:
        - DataFrame: DataFrame containing mock demand data.
        """
        mock_elasticity_list = []
        mock_elasticity_list.append(self.generate_linear_elasticity())
        mock_elasticity_list.append(self.generate_nonlinear_elasticity())

        return pd.concat(mock_elasticity_list)


if __name__ == "__main__":
    mock_elasticity_generator = MockElasticityGenerator()
    print(mock_elasticity_generator.generate_mock_elasticity_data())
