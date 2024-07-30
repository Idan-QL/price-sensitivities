"""utils for calculating quantity demanded and price elasticity of demand."""

import numpy as np
import pandas as pd


def normalised_rmse(
    model_type: str,
    price_col: str,
    quantity_col: str,
    data_test: pd.DataFrame,
    a_linear: float,
    b_linear: float,
) -> float:
    """Calculate the Normalised Root Mean Squared Error (RMSE) by median.

    Args:
        model_type (str): The type of model being used.
        price_col (str): The column name for the price data.
        quantity_col (str): The column name for the quantity data.
        data_test (pd.DataFrame): The test dataset containing price and quantity data.
        a_linear (float): The coefficient 'a_linear' used in the quantity calculation.
        b_linear (float): The coefficient 'b_linear' used in the quantity calculation.

    Returns:
        float: The normalised RMSE value.

    """
    predicted_quantity = np.vectorize(
        lambda p: calculate_quantity_from_price(p, a_linear, b_linear, model_type)
    )(data_test[price_col])
    mean_squared_error = np.mean((data_test[quantity_col] - predicted_quantity) ** 2)
    rmse = np.sqrt(mean_squared_error)
    median_predicted_quantity = np.median(predicted_quantity)
    return round(rmse / median_predicted_quantity, 4)


def relative_absolute_error_calculation(
    model_type: str,
    price_col: str,
    quantity_col: str,
    data_test: pd.DataFrame,
    a_linear: float,
    b_linear: float,
) -> float:
    """Calculate the relative absolute error for a given model normalised by the max.

    Args:
        model_type (str): The type of the model.
        price_col (str): The name of the column containing the prices.
        quantity_col (str): The name of the column containing the quantities.
        data_test (pandas.DataFrame): The test dataset.
        a_linear (float): The coefficient 'a_linear' used in the quantity calculation.
        b_linear (float): The coefficient 'b_linear' used in the quantity calculation.

    Returns:
    - relative_absolute_error (float): The relative absolute error.
    """
    predicted_quantity = np.vectorize(
        lambda p: calculate_quantity_from_price(p, a_linear, b_linear, model_type)
    )(data_test[price_col])

    absolute_errors = np.abs(data_test[quantity_col] - predicted_quantity)
    normaliser = np.maximum.reduce(
        [np.abs(data_test[quantity_col].values), np.abs(predicted_quantity)]
    )
    return np.mean(absolute_errors / normaliser) * 100


def calculate_quantity_from_price(
    price: float, a_linear: float, b_linear: float, model_type: str
) -> float:
    """Calculate quantity demanded given price and demand model parameters.

    Args:
        price (float): Price of the product.
        a_linear (float): Intercept parameter of the demand model.
        b_linear (float): Slope parameter of the demand model.
        model_type (str): Type of demand model ('linear', 'power', or 'exponential').

    Returns:
        float: Quantity demanded.
    """
    if model_type == "linear":
        return linear_demand(price, a_linear, b_linear)
    if model_type == "power":
        return power_demand(price, a_linear, b_linear)
    if model_type == "exponential":
        return exponential_demand(price, a_linear, b_linear)
    raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")


def linear_demand(price: float, a_linear: float, b_linear: float) -> float:
    """Linear demand model: Q = a + bP.

    Args:
        price (float): The price of the product.
        a_linear (float): The intercept of the demand curve.
        b_linear (float): The slope of the demand curve.

    Returns:
        float: The quantity demanded at the given price.

    """
    return a_linear + b_linear * price


def power_demand(price: float, a_linear: float, b_linear: float) -> float:
    """Power demand model: Q = a * P**b.

    a_linear and b_linear are the result of a fit after linear transformation
    log(Q) = log(a_power) + b_power * log(P)
    log(Q) = a_linear + b_linear * log(P)

    Args:
        price (float): Price.
        a_linear (float): The coefficient a_linear.
        b_linear (float): The coefficient b_linear.

    Returns:
    - float: The calculated power demand.

    """
    log_price = np.log(price)
    log_q = linear_demand(log_price, a_linear, b_linear)
    return np.exp(log_q)


def exponential_demand(price: float, a_linear: float, b_linear: float) -> float:
    """Exponential demand model: Q = a * exp(-bP).

    a_linear and b_linear are the result of a fit after linear transformation
    log(Q) = log(a_exp) +  b_exp * P
    log(Q) = a_linear + b_linear * P
    Args:
        price (float): The price of the product.
        a_linear (float): The coefficient a_linear.
        b_linear (float): The coefficient b_linear.

    Returns:
    - float: The quantity demanded according to the exponential demand model.
    """
    log_q = linear_demand(price, a_linear, b_linear)
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

    def partial_derivative_a(a: float, b: float, p: float) -> float:
        return (-b * p) / (a + b * p) ** 2

    def partial_derivative_b(a: float, b: float, p: float) -> float:
        return (a * p) / (a + b * p) ** 2

    # Calculate partial derivatives at the given price p
    partial_a = partial_derivative_a(a, b, p)
    partial_b = partial_derivative_b(a, b, p)

    # Extract the covariance matrix elements
    cov_aa = cov_matrix.iloc[0, 0]
    cov_bb = cov_matrix.iloc[1, 1]
    cov_ab = cov_matrix.iloc[0, 1]

    # Calculate the variance of elasticity using the covariance matrix and partial derivatives
    variance = partial_a**2 * cov_aa + partial_b**2 * cov_bb + 2 * partial_a * partial_b * cov_ab

    # Return the standard deviation (square root of the variance)
    return np.sqrt(variance)


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
