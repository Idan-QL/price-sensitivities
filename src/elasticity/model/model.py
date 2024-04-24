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
    elasticity_error_propagation = calculate_elasticity_error_propagation(
        model_type,
        model.params.iloc[0],
        model.params.iloc[1],
        cov_matrix,
        median_price)
    a, b = model.params.iloc[0], model.params.iloc[1]
    elasticity = calculate_elasticity_from_parameters(model_type, a, b, median_price)
    return a, b, pvalue, r_squared, elasticity, elasticity_error_propagation


def calculate_elasticity_error_propagation(model_type: str,
                                           a: float,
                                           b: float,
                                           cov_matrix: np.ndarray,
                                           p: float) -> float:
    """Calculate the standard errors of the elasticity function."""
    if model_type == "linear":
        return elasticity_error_propagation_linear(a, b, cov_matrix, p)
    elif model_type == "exponential":
        return elasticity_error_propagation_exponential(cov_matrix, p)
    elif model_type == "power":
        return elasticity_error_propagation_power(cov_matrix)


def elasticity_error_propagation_linear(a: float, 
                                        b: float, 
                                        cov_matrix: np.ndarray, 
                                        p: float) -> float:
    """
    Calculate the errors of the elasticity function for a linear model.

    Parameters:
    - a: Estimated parameter for 'a'
    - b: Estimated parameter for 'b'
    - cov_matrix: Covariance matrix of the parameter estimates
    - p: Price

    Returns:
    - float: Standard deviation of elasticity at price p
    """

    # Calculate the partial derivatives of the elasticity function
    def partial_derivative_a_linear(a: float, b: float, x: float) -> float:
        return -b * x / (a - b * x)**2

    def partial_derivative_b_linear(a: float, b: float, x: float) -> float:
        return x / (a - b * x)
    
    a_label = cov_matrix.columns[0]
    b_label = cov_matrix.columns[1]

    return np.sqrt((partial_derivative_a_linear(a, b, p)**2 * cov_matrix[a_label][a_label]) + 
                   (partial_derivative_b_linear(a, b, p)**2 * cov_matrix[b_label][b_label]) + 
                   (2 * partial_derivative_a_linear(a, b, p) *
                    partial_derivative_b_linear(a, b, p) *
                    cov_matrix[a_label][b_label]))


def elasticity_error_propagation_exponential(cov_matrix: np.ndarray, 
                                             p: float) -> float:
    """
    Calculate the standard errors of the elasticity function for an exponential model.

    Parameters:
    - cov_matrix: Covariance matrix of the parameter estimates
    - p: Price

    partial_derivative_b_exponential(b: float, x: np.ndarray) -> np.ndarray:
        return x

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function at price p
    """

    b_label = cov_matrix.columns[1]
    return np.sqrt((p**2) * cov_matrix[b_label][b_label])


def elasticity_error_propagation_power(cov_matrix: np.ndarray) -> float:
    """
    Calculate the standard errors of the elasticity function for a power model.

    Parameters:
    - cov_matrix: Covariance matrix of the parameter estimates

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function
    """

    b_label = cov_matrix.columns[1]
    return np.sqrt(cov_matrix[b_label][b_label])
