"""utils for calculating quantity demanded and price elasticity of demand."""

import numpy as np


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
    """Linear demand model: Q = a + bP."""
    return a + b * price


def power_demand(price: float, a: float, b: float) -> float:
    """Power demand model also called constant elasticity: Q = a * P**b."""
    log_price = np.log(price)
    log_q = linear_demand(log_price, a, b)  # return log Q
    return np.exp(log_q)


def exponential_demand(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP)."""
    log_q = linear_demand(price, a, b)
    return np.exp(log_q)


def linear_elasticity(price: float, a: float, b: float) -> float:
    """Linear demand model elasticity: e = -bP / (a + bP)."""
    demand = linear_demand(price, a, b)
    if demand == 0:
        return float("inf") if price == 0 else float("-inf")
    return b * price / demand


def power_elasticity(b: float) -> float:
    """Power demand model elasticity: e = b."""
    return b


def exponential_elasticity(price: float, b: float) -> float:
    """Exponential demand model elasticity: e = b * price."""
    return b * price


def calculate_elasticity_from_parameters(
    model_type: str, a: float, b: float, price: float
) -> float:
    """Calculate price elasticity of demand (PED) given coefficients and price points."""
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

    Parameters:
    - a: Estimated parameter for 'a'
    - b: Estimated parameter for 'b'
    - cov_matrix: Covariance matrix of the parameter estimates
    - p: Price

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

    Parameters:
    - a: Estimated parameter for 'a'
    - b: Estimated parameter for 'b'
    - cov_matrix: Covariance matrix of the parameter estimates
    - p: Price

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

    Parameters:
    - cov_matrix: Covariance matrix of the parameter estimates
    - p: price

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function at price p
    """
    b_label = cov_matrix.columns[1]
    return np.sqrt((p**2) * cov_matrix[b_label][b_label])


def elasticity_error_propagation_power(cov_matrix: np.ndarray) -> float:
    """Calculate the standard errors of the elasticity function for a power model.

    Parameters:
    - cov_matrix: Covariance matrix of the parameter estimates

    Returns:
    - std_errors_elasticity: Standard errors of the elasticity function
    """
    b_label = cov_matrix.columns[1]
    return np.sqrt(cov_matrix[b_label][b_label])
