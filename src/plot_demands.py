"""Module of plot_demands."""

import matplotlib.pyplot as plt
import numpy as np


def plot_demand_curves() -> None:
    """plot_demand_curves."""
    def linear_demand_equation(price: float,
                               a: float,
                               b: float) -> float:
        """Linear demand model: Q = a - bP."""
        return a + b * price

    def exponential_demand_equation(price: float,
                                    a: float,
                                    b: float) -> float:
        """Exponential demand model: Q = a * exp(-bP)."""
        return a * np.exp(b * price)

    def power_demand_equation(price: float,
                              a: float,
                              b: float) -> float:
        """Power demand model: Q = a * P^(-b)."""
        return a * (price ** (b))

    # Define parameters
    a_linear, b_linear = 10, -0.1
    a_exponential, b_exponential = 30, -0.03
    a_power, b_power = 1000000, -3

    # Generate price points
    prices = np.linspace(50, 100, 100)

    # Calculate quantity demanded for each price
    quantities_linear = linear_demand_equation(prices, a_linear, b_linear)
    quantities_exponential = exponential_demand_equation(prices, a_exponential, b_exponential)
    quantities_power = power_demand_equation(prices, a_power, b_power)

    # Plot demand curves with different colors
    plt.plot(quantities_linear, prices, label='Linear Demand - Q = 10 + -0.1xP', color='blue')
    plt.plot(quantities_exponential, prices, color='red',
             label='Exponential Demand - Q = 30 * exp(-0.03xP) - Elasticity = -0.03*P')
    plt.plot(quantities_power, prices, color='green',
             label='Power Demand - Q = 1000000 * P^(-3) - Constant Elasticity = -3')

    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.title('Demand Curves')
    plt.grid(True)

    # Place legend outside the plot at the bottom
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()

# Example usage:
# plot_demand_curves()
