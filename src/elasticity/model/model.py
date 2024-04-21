"""Module of modeling."""

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from elasticity.model.utils import calculate_elasticity_from_parameters


def estimate_coefficients(
    data: pd.DataFrame,
    model_type: str,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
) -> Tuple[float, float, float, float, float]:
    """Estimate coefficients for demand model using log transformation if nonlinear."""
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
    a, b = model.params.iloc[0], model.params.iloc[1]
    elasticity = calculate_elasticity_from_parameters(model_type, a, b, median_price)
    return a, b, pvalue, r_squared, elasticity, cov_matrix


def elasticity_error_propagation_linear(a: float, 
                                        b: float, 
                                        cov_matrix: np.ndarray, 
                                        x: np.ndarray) -> float:
    """
    Calculate the standard errors of the elasticity function for a linear model.

    Parameters:
    - a: Estimated parameter for 'a'
    - b: Estimated parameter for 'b'
    - cov_matrix: Covariance matrix of the parameter estimates
    - x: Independent variable values

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function
    """

    # Calculate the partial derivatives of the elasticity function
    def partial_derivative_a_linear(a: float, b: float, x: float) -> float:
        return -b * x / (a - b * x)**2

    def partial_derivative_b_linear(a: float, b: float, x: float) -> float:
        return x / (a - b * x)

    # Calculate the standard errors of the elasticity function
    std_errors_elasticity = np.sqrt((partial_derivative_a_linear(a, b, x)**2 * cov_matrix[0, 0]) + 
                                    (partial_derivative_b_linear(a, b, x)**2 * cov_matrix[1, 1]) + 
                                    2 * partial_derivative_a_linear(a, b, x) * partial_derivative_b_linear(a, b, x) * cov_matrix[0, 1])

    return std_errors_elasticity


def elasticity_error_propagation_exponential(b: float, 
                                             cov_matrix: np.ndarray, 
                                             x: np.ndarray) -> float:
    """
    Calculate the standard errors of the elasticity function for an exponential model.

    Parameters:
    - b: Estimated parameter for 'b'
    - cov_matrix: Covariance matrix of the parameter estimates
    - x: Independent variable values

    partial_derivative_b_exponential(b: float, x: np.ndarray) -> np.ndarray:
        return x

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function
    """

    # Calculate the standard errors of the elasticity function
    std_errors_elasticity = np.sqrt((x**2) * cov_matrix[0, 0])

    return std_errors_elasticity


def elasticity_error_propagation_power(b: float, 
                                       cov_matrix: np.ndarray) -> float:
    """
    Calculate the standard errors of the elasticity function for a power model.

    Parameters:
    - b: Estimated parameter for 'b'
    - cov_matrix: Covariance matrix of the parameter estimates

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function
    """

    # Calculate the standard errors of the elasticity function
    std_errors_elasticity = np.sqrt(cov_matrix[0, 0])

    return std_errors_elasticity
