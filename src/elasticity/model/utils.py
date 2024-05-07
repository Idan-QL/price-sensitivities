"""utils for calculating quantity demanded and price elasticity of demand."""

import numpy as np
import pandas as pd


def relative_absolute_error_calculation(
    model_type: str,
    price_col: str,
    quantity_col: str,
    data_test: pd.DataFrame,
    a: float,
    b: float,
) -> float:
    """Calculate the relative absolute error for a given model.

    Args:
        model_type (str): The type of the model.
        price_col (str): The name of the column containing the prices.
        quantity_col (str): The name of the column containing the quantities.
        data_test (pandas.DataFrame): The test dataset.
        a (float): The coefficient 'a' for the model.
        b (float): The coefficient 'b' for the model.

    Returns:
    - relative_absolute_error (float): The relative absolute error.
    """
    predicted_quantity = [
        calculate_quantity_from_price(p, a, b, model_type) for p in data_test[price_col]
    ]
    absolute_errors = np.abs(data_test[quantity_col] - predicted_quantity)

    return np.mean(absolute_errors / data_test[quantity_col]) * 100


def calculate_quantity_from_price(
    price: float, a: float, b: float, model_type: str
) -> float:
    """Calculate quantity demanded given price and demand model parameters.

    Args:
        price (float): Price of the product.
        a (float): Intercept parameter of the demand model.
        b (float): Slope parameter of the demand model.
        model_type (str): Type of demand model ('linear', 'power', or 'exponential').

    Returns:
        float: Quantity demanded.
    """
    if model_type == "linear":
        return linear_demand(price, a, b)
    if model_type == "power":
        return power_demand(price, a, b)
    if model_type == "exponential":
        return exponential_demand(price, a, b)
    raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")


def linear_demand(price: float, a: float, b: float) -> float:
    """Linear demand model: Q = a + bP.

    Args:
        price (float): The price of the product.
        a (float): The intercept of the demand curve.
        b (float): The slope of the demand curve.

    Returns:
        float: The quantity demanded at the given price.

    """
    return a + b * price


def power_demand(price: float, a: float, b: float) -> float:
    """Power demand model: Q = a * P**b.

    Args:
        price (float): The price of the power.
        a (float): The coefficient a in the power demand model.
        b (float): The coefficient b in the power demand model.

    Returns:
    - float: The calculated power demand.

    """
    log_price = np.log(price)
    log_q = linear_demand(log_price, a, b)  # return log Q
    return np.exp(log_q)


def exponential_demand(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP).

    Args:
        price (float): The price of the product.
        a (float): The coefficient 'a' in the demand equation.
        b (float): The coefficient 'b' in the demand equation.

    Returns:
    - float: The quantity demanded according to the exponential demand model.
    """
    log_q = linear_demand(price, a, b)
    return np.exp(log_q)


def linear_elasticity(price: float, a: float, b: float) -> float:
    """Calculate the linear demand model elasticity.

    The elasticity is calculated using the formula: e = -bP / (a + bP),
    where e is the elasticity, P is the price, a is a constant, and b is a constant.

    Args:
        price (float): The price of the product.
        a (float): The constant a in the demand model.
        b (float): The constant b in the demand model.

    Returns:
        float: The elasticity value.

    Raises:
        ZeroDivisionError: If the demand is zero and the price is non-zero.

    """
    demand = linear_demand(price, a, b)
    if demand == 0:
        return float("inf") if price == 0 else float("-inf")
    return b * price / demand


def power_elasticity(b: float) -> float:
    """Calculate the power demand model elasticity.

    Args:
        b (float): The elasticity value.

    Returns:
    - float: The elasticity value.

    """
    return b


def exponential_elasticity(price: float, b: float) -> float:
    """Exponential demand model elasticity: e = b * price.

    Args:
        price (float): The price of the product.
        b (float): The elasticity coefficient.

    Returns:
        float: The elasticity value.

    """
    return b * price


def calculate_elasticity_from_parameters(
    model_type: str, a: float, b: float, price: float
) -> float:
    """Calculate price elasticity of demand (PED) given coefficients and price points.

    Args:
        model_type (str): The type of model to use for calculating elasticity.
        Valid options are 'linear', 'power', or 'exponential'.
        a (float): The coefficient 'a' used in the linear elasticity model.
        b (float): The coefficient 'b' used in the linear, power, or exponential elasticity model.
        price (float): The price point at which to calculate the elasticity.

    Returns:
        float: The calculated price elasticity of demand (PED) rounded to 2 decimal places.

    Raises:
        ValueError: If an invalid model type is provided.

    """
    if model_type == "linear":
        elasticity = linear_elasticity(price, a, b)
    elif model_type == "power":
        elasticity = power_elasticity(b)
    elif model_type == "exponential":
        elasticity = exponential_elasticity(price, b)
    else:
        raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")
    return round(elasticity, 2)


def calculate_elasticity_error_propagation(
    model_type: str, a: float, b: float, cov_matrix: np.ndarray, p: float
) -> float:
    """Calculates the standard error of the elasticity function based on the model type.

    Args:
        model_type (str): The type of model to use for calculating elasticity.
        a: Estimated parameter for 'a'
        b: Estimated parameter for 'b'
        cov_matrix: Covariance matrix of the parameter estimates
        p: Price

    Returns:
        float: The standard error of the elasticity function.

    Raises:
        ValueError: If an unsupported model type is provided.
    """
    if model_type == "linear":
        return elasticity_error_propagation_linear(a, b, cov_matrix, p)
    if model_type == "exponential":
        return elasticity_error_propagation_exponential(cov_matrix, p)
    if model_type == "power":
        return elasticity_error_propagation_power(cov_matrix)
    raise ValueError(f"Unsupported model type: {model_type}")


def elasticity_error_propagation_linear(
    a: float, b: float, cov_matrix: np.ndarray, p: float
) -> float:
    """Calculate the errors of the elasticity function for a linear model.

    Args:
        a: Estimated parameter for 'a'
        b: Estimated parameter for 'b'
        cov_matrix: Covariance matrix of the parameter estimates
        p: Price

    Returns:
    - float: Standard deviation of elasticity at price p
    """

    def partial_derivative_a_linear(a: float, b: float, p: float) -> float:
        return (-b * p) / (a + b * p) ** 2

    def partial_derivative_b_linear(a: float, b: float, p: float) -> float:
        return (a * p) / (a + b * p) ** 2

    a_label = cov_matrix.columns[0]
    b_label = cov_matrix.columns[1]

    return np.sqrt(
        (partial_derivative_a_linear(a, b, p) ** 2 * cov_matrix[a_label][a_label])
        + (partial_derivative_b_linear(a, b, p) ** 2 * cov_matrix[b_label][b_label])
        + (
            2
            * partial_derivative_a_linear(a, b, p)
            * partial_derivative_b_linear(a, b, p)
            * cov_matrix[a_label][b_label]
        )
    )


def elasticity_error_propagation_exponential(cov_matrix: np.ndarray, p: float) -> float:
    """Calculate the standard errors of the elasticity function for an exponential model.

    Args:
        cov_matrix: Covariance matrix of the parameter estimates
        p: price

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function at price p
    """
    b_label = cov_matrix.columns[1]
    return np.sqrt((p**2) * cov_matrix[b_label][b_label])


def elasticity_error_propagation_power(cov_matrix: np.ndarray) -> float:
    """Calculate the standard errors of the elasticity function for a power model.

    Args:
        cov_matrix: Covariance matrix of the parameter estimates

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function
    """
    b_label = cov_matrix.columns[1]
    return np.sqrt(cov_matrix[b_label][b_label])
