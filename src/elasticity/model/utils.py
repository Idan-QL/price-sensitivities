"""utils for calculating quantity demanded and price elasticity of demand."""
import numpy as np


def calculate_quantity_from_price(price: float, a: float, b: float, model_type: str) -> float:
    """Calculate quantity demanded given price and demand model parameters.

    Args:
        price (float): Price of the product.
        a (float): Intercept parameter of the demand model.
        b (float): Slope parameter of the demand model.
        model_type (str): Type of demand model ('linear', 'power', or 'exponential').

    Returns:
        float: Quantity demanded.
    """
    if model_type == 'linear':
        return linear_demand(price, a, b)
    if model_type == 'power':
        return power_demand(price, a, b)
    if model_type == 'exponential':
        return exponential_demand(price, a, b)
    raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")

def linear_demand(price: float, a: float, b: float) -> float:
    """Linear demand model: Q = a + bP."""
    return a + b * price

def power_demand(price: float, a: float, b: float) -> float:
    """Power demand model also called constant elasticity: Q = a * P**b."""
    log_price = np.log(price)
    log_q = linear_demand(log_price, a, b) # return log Q
    return np.exp(log_q)

def exponential_demand(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP)."""
    log_q = linear_demand(price, a, b)
    return np.exp(log_q)

def linear_elasticity(price: float, a: float, b: float) -> float:
    """Linear demand model elasticity: e = -bP / (a + bP)."""
    demand = linear_demand(price, a, b)
    if demand == 0:
        return float('inf') if price == 0 else float('-inf')
    return b * price / demand

def power_elasticity(b: float) -> float:
    """Power demand model elasticity: e = b."""
    return b

def exponential_elasticity(price: float, b: float) -> float:
    """Exponential demand model elasticity: e = b * price."""
    return b * price

def calculate_elasticity_from_parameters(model_type: str,
                                         a: float,
                                         b: float,
                                         price: float) -> float:
    """Calculate price elasticity of demand (PED) given coefficients and price points."""
    if model_type == 'linear' :
        elasticity = linear_elasticity(price, a, b)
    elif model_type == 'power':
        elasticity = power_elasticity(b)
    elif model_type == 'exponential':
        elasticity = exponential_elasticity(price, b)
    else:
        raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")
    return round(elasticity, 2)
