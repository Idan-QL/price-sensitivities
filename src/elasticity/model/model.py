"""Module of modeling."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel

from elasticity.model.utils import (
    calculate_elasticity_error_propagation,
    calculate_elasticity_from_parameters,
    normalised_rmse,
    relative_absolute_error_calculation,
)


class EstimationResult(BaseModel):
    """Represents the result of coefficient estimation for a demand model.

    Attributes:
        a (float): The estimated coefficient 'a'.
        b (float): The estimated coefficient 'b'.
        pvalue (float): The p-value associated with the estimated coefficients.
        r2 (float): The R-squared value of the fitted model.
        elasticity (float): The elasticity calculated from the estimated coefficients.
        elasticity_error_propagation (float): The standard deviation of elasticity.
        aic (float): The Akaike Information Criterion (AIC) of the fitted model.
        relative_absolute_error (float): The relative absolute error of the fitted model.
        norm_rmse (float): The normalized root mean squared error of the fitted model.
    """

    a: float = float("inf")
    b: float = float("inf")
    pvalue: float = float("inf")
    r2: float = float("inf")
    elasticity: float = float("inf")
    elasticity_error_propagation: float = float("inf")
    aic: float = float("inf")
    relative_absolute_error: float = float("inf")
    norm_rmse: float = float("inf")


def validate_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate the input data for coefficient estimation.

    Args:
        data (pd.DataFrame): The input data containing the required columns.
        required_columns (list[str]): A list of column names that must be present in the data.

    Returns:
        bool: True if the data is valid, False otherwise.
    """
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logging.warning(f"Missing required columns: {', '.join(missing_columns)}")
        return False
    if data.shape[0] < 3:
        logging.warning("Insufficient data to fit the model. Need at least 3 rows of data.")
        return False
    return True


def check_positive_values(data: pd.DataFrame, columns: List[str]) -> bool:
    """Check if specified columns contain only positive values.

    Args:
        data (pd.DataFrame): The input data.
        columns (list[str]): A list of column names to check.

    Returns:
        bool: True if all specified columns contain only positive values, False otherwise.
    """
    for column in columns:
        if (data[column] <= 0).any():
            logging.warning(
                f"Column '{column}' contains negative values, not suitable for log transformation."
            )
            return False
    return True


def fit_model(
    data: pd.DataFrame,
    model_type: str,
    price_col: str,
    quantity_col: str,
    weights_col: str,
    competitor_price_col: Optional[str] = None,
) -> sm.WLS:
    """Fit a weighted least squares (WLS) model.

    with an optional avg_competitor_price as a covariate.

    Args:
        data (pd.DataFrame): The input data.
        model_type (str): The type of demand model to fit.
        price_col (str): The name of the column representing price.
        quantity_col (str): The name of the column representing quantity.
        weights_col (str): The name of the column representing weights.
        competitor_price_col (str, optional): The name of the column representing competitor price.
                                              Defaults to None.

    Returns:
        sm.WLS: The fitted WLS model.
    """
    # Include competitor price column if provided
    if competitor_price_col:
        x = sm.add_constant(data[[price_col, competitor_price_col]])
    else:
        x = sm.add_constant(data[[price_col]])

    y = data[quantity_col]
    weights = data[weights_col]

    # Apply log transformations for specific model types
    if model_type == "power":
        y = np.log(y)
        x[price_col] = np.log(x[price_col])
        if competitor_price_col:
            x[competitor_price_col] = np.log(x[competitor_price_col])
    elif model_type == "exponential":
        y = np.log(y)

    # Fit WLS model
    return sm.WLS(y, x, weights=weights).fit()


def estimate_coefficients(
    data: pd.DataFrame,
    model_type: str,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
    competitor_price_col: Optional[str] = None,
) -> EstimationResult:
    """Estimate coefficients for demand model using log transformation if nonlinear.

    with an optional avg_competitor_price as a covariate.

    Args:
        data (pd.DataFrame): The input data containing the price and quantity columns.
        model_type (str): The type of demand model to use.
                          Valid options are "power" and "exponential".
        price_col (str, optional): The name of the column in the data frame
                                   that represents the price. Defaults to "price".
        quantity_col (str, optional): The name of the column in the data frame
                                      that represents the quantity. Defaults to "quantity".
        weights_col (str, optional): The name of the column in the data frame
                                     that represents the weights. Defaults to "days".
        competitor_price_col (str, optional): The name of the column representing the
                                              competitor price. Defaults to None.

    Returns:
        EstimationResult: An instance of EstimationResult containing the estimated coefficients
        and metrics. If an error occurs, returns an instance of EstimationResult
        with default values (all Inf).
    """
    try:
        # Validate the input data
        required_cols = [price_col, quantity_col, weights_col]
        if competitor_price_col:
            required_cols.append(competitor_price_col)

        if not validate_data(data, required_cols):
            return EstimationResult()
        if not check_positive_values(
            data,
            [price_col, quantity_col] + ([competitor_price_col] if competitor_price_col else []),
        ):
            return EstimationResult()

        # Fit the model with or without competitor price
        model = fit_model(
            data, model_type, price_col, quantity_col, weights_col, competitor_price_col
        )

        # Extract metrics and perform calculations
        pvalue = model.f_pvalue
        r2 = model.rsquared
        cov_matrix = model.cov_params()
        last_price = data["last_price"].median()

        elasticity_error_propagation = calculate_elasticity_error_propagation(
            model_type,
            model.params.iloc[0],
            model.params.iloc[1] if len(model.params) > 1 else None,
            cov_matrix,
            last_price,
        )
        a, b = model.params.iloc[0], model.params.iloc[1] if len(model.params) > 1 else None
        aic = model.aic
        elasticity = calculate_elasticity_from_parameters(model_type, a, b, last_price)
        relative_absolute_error = relative_absolute_error_calculation(
            model_type, price_col, quantity_col, data, a, b
        )
        norm_rmse = normalised_rmse(model_type, price_col, quantity_col, data, a, b)

        # Return results as an instance of EstimationResult
        return EstimationResult(
            a=a,
            b=b,
            pvalue=pvalue,
            r2=r2,
            elasticity=elasticity,
            elasticity_error_propagation=elasticity_error_propagation,
            aic=aic,
            relative_absolute_error=relative_absolute_error,
            norm_rmse=norm_rmse,
        )
    except Exception as e:
        logging.error(f"Error during model estimation: {e}")
        return EstimationResult()


# Example usage:
# data = pd.DataFrame({
#     "price": [10, 20, 30],
#     "quantity": [100, 150, 200],
#     "days": [1, 1, 1]
# })
# result = estimate_coefficients(data, "power")
# print(result)
