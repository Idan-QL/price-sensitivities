"""Modul of cross-validation function."""

import logging
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from elasticity.model.model import estimate_coefficients
from elasticity.model.utils import normalised_rmse, relative_absolute_error_calculation


class CrossValidationResult(BaseModel):
    """Represents the result of cross_validation.

    Attributes:
        mean_relative_absolute_error (float): The estimated coefficient 'a'.
        mean_norm_rmse (float): The estimated coefficient 'b'.
        mean_a (float): The p-value associated with the estimated coefficients.
        mean_b (float): The R-squared value of the fitted model.
        mean_elasticity (float): The elasticity calculated from the estimated coefficients.
        mean_r2 (float): The standard deviation of elasticity.
        aic (float): The Akaike Information Criterion (AIC) of the fitted model.
        relative_absolute_error (float): The relative absolute error of the fitted model.
        norm_rmse (float): The normalized root mean squared error of the fitted model.
    """

    mean_a: float = float("inf")
    mean_b: float = float("inf")
    mean_elasticity: float = float("inf")
    mean_relative_absolute_error: float = float("inf")
    mean_norm_rmse: float = float("inf")
    mean_r2: float = float("inf")


def cross_validation(
    data: pd.DataFrame,
    model_type: str,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
    n_tests: int = 3,
) -> CrossValidationResult:
    """Perform cross-validation.

    Args:
        data (pd.DataFrame): The input data for cross-validation.
        model_type (str): The type of model to use for estimation.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.1.
        price_col (str, optional): The name of the column containing prices. Defaults to "price".
        quantity_col (str, optional): The name of the column containing quantities.
        Defaults to "quantity".
        weights_col (str, optional): The name of the column containing weights. Defaults to "days".
        n_tests (int, optional): The number of cross-validation tests to perform. Defaults to 3.

    Returns:
        CrossValidationResult: An instance of CrossValidationResult containing the mean relative
        absolute error, mean normalized RMSE, mean coefficient 'a', mean coefficient 'b',
        mean elasticity, and mean R-squared value.
        If an error occurs, returns an instance of CrossValidationResult with default
        values (all Inf).
    """
    relative_absolute_errors_test: List[float] = []
    normalised_rmse_test: List[float] = []
    a_lists: List[float] = []
    b_lists: List[float] = []
    elasticity_lists: List[float] = []
    r2_lists: List[float] = []

    try:
        for i in range(n_tests):
            data_train, data_test = train_test_split(data, test_size=test_size, random_state=42 + i)

            result = estimate_coefficients(
                data_train,
                model_type,
                price_col=price_col,
                quantity_col=quantity_col,
                weights_col=weights_col,
            )

            if any(value == float("inf") for value in result.model_dump().values()):
                logging.warning(f"Estimation failed in iteration {i}.")
                continue

            a_lists.append(result.a)
            b_lists.append(result.b)
            elasticity_lists.append(result.elasticity)
            r2_lists.append(result.r2)

            relative_absolute_errors_test.append(
                relative_absolute_error_calculation(
                    model_type, price_col, quantity_col, data_test, result.a, result.b
                )
            )

            normalised_rmse_test.append(
                normalised_rmse(model_type, price_col, quantity_col, data_test, result.a, result.b)
            )

        if not a_lists:
            logging.warning("Cross-validation failed: No valid results.")
            return CrossValidationResult()

        mean_relative_absolute_error = np.mean(relative_absolute_errors_test)
        mean_norm_rmse = np.mean(normalised_rmse_test)
        mean_a = np.mean(a_lists)
        mean_b = np.mean(b_lists)
        mean_elasticity = np.mean(elasticity_lists)
        mean_r2 = np.mean(r2_lists)

        return CrossValidationResult(
            mean_a=mean_a,
            mean_b=mean_b,
            mean_elasticity=mean_elasticity,
            mean_relative_absolute_error=mean_relative_absolute_error,
            mean_norm_rmse=mean_norm_rmse,
            mean_r2=mean_r2,
        )

    except Exception as e:
        logging.error(f"An error occurred during cross-validation: {e}")
        return CrossValidationResult()
