"""Module of modeling."""

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from elasticity.model.utils import (
    calculate_elasticity_error_propagation,
    calculate_elasticity_from_parameters,
    relative_absolute_error_calculation
)


def estimate_coefficients(
    data: pd.DataFrame,
    model_type: str,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
) -> Tuple[float, float, float, float, float]:
    """
    Estimate coefficients for demand model using log transformation if nonlinear.

    Parameters:
        data (pd.DataFrame): The input data containing the price and quantity columns.
        model_type (str): The type of demand model to use. Valid options are "power" and "exponential".
        price_col (str, optional): The name of the column in the data frame that represents the price. Defaults to "price".
        quantity_col (str, optional): The name of the column in the data frame that represents the quantity. Defaults to "quantity".
        weights_col (str, optional): The name of the column in the data frame that represents the weights. Defaults to "days".

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the estimated coefficients, p-value, R-squared value,
        elasticity, and other relevant metrics.
    """
    X = sm.add_constant(data[[price_col]])
    y = data[quantity_col]
    weights = data[weights_col]

    if model_type == "power":
        y = np.log(y)
        X[price_col] = np.log(X[price_col])
    elif model_type == "exponential":
        y = np.log(y)

    model = sm.WLS(y, X, weights=weights).fit()
    pvalue = model.f_pvalue
    r_squared = model.rsquared
    cov_matrix = model.cov_params()
    median_price = data[price_col].median()
    elasticity_error_propagation = calculate_elasticity_error_propagation(
        model_type, model.params.iloc[0], model.params.iloc[1], cov_matrix, median_price
    )
    a, b = model.params.iloc[0], model.params.iloc[1]
    aic = model.aic
    elasticity = calculate_elasticity_from_parameters(model_type, a, b, median_price)
    relative_absolute_error = relative_absolute_error_calculation(model_type,
                                                                  price_col,
                                                                  quantity_col,
                                                                  data,
                                                                  a,
                                                                  b)
    return (a,
            b,
            pvalue,
            r_squared,
            elasticity,
            elasticity_error_propagation,
            aic,
            relative_absolute_error)
