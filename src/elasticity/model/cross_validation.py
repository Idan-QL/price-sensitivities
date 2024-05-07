"""Modul of cross-validation function."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from elasticity.model.model import estimate_coefficients
from elasticity.model.utils import relative_absolute_error_calculation


def cross_validation(
    data: pd.DataFrame,
    model_type: str,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
    n_tests: int = 3,
) -> Tuple[float, float, float, float, float]:
    """Perform cross-validation."""
    relative_absolute_errors_test = []
    a_lists = []
    b_lists = []
    elasticity_lists = []
    r_squared_lists = []
    for i in range(n_tests):
        data_train, data_test = train_test_split(
            data, test_size=test_size, random_state=42 + i
        )
        a, b, _, r_squared, elasticity, _, _, _= estimate_coefficients(
            data_train,
            model_type,
            price_col=price_col,
            quantity_col=quantity_col,
            weights_col=weights_col,
        )

        relative_absolute_errors_test.append(
            relative_absolute_error_calculation(
                model_type,
                price_col,
                quantity_col,
                data_test,
                a,
                b))
        a_lists.append(a)
        b_lists.append(b)
        elasticity_lists.append(elasticity)
        r_squared_lists.append(r_squared)

    # Return the average relative absolute error
    mean_relative_error_test = np.mean(relative_absolute_errors_test)
    mean_a = np.mean(a_lists)
    mean_b = np.mean(b_lists)
    mean_elasticity = np.mean(elasticity_lists)
    mean_r_squared = np.mean(r_squared_lists)

    return mean_relative_error_test, mean_a, mean_b, mean_elasticity, mean_r_squared

