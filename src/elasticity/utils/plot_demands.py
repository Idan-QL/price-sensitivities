"""Module of plot_demands."""

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


def linear_demand_equation(price: float, a: float, b: float) -> float:
    """Linear demand model: Q = a - bP."""
    return a + b * price

def exponential_demand_equation(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP)."""
    return a * np.exp(b * price)

def power_demand_equation(price: float, a: float, b: float) -> float:
    """Power demand model: Q = a * P^(-b)."""
    return a * (price ** (b))

def plot_model_and_prices(df_results, df_data, uid, title='', price_col='round_price')-> None:
    """plot_demand_curves from model result and actual data."""
    print('uid:', uid)
    df_results_uid = df_results[df_results.uid==uid]
    df_uid = df_data[df_data.uid==uid]
    a_linear=df_results_uid['best_model_a'].iloc[0]
    b_linear=df_results_uid['best_model_b'].iloc[0]
    model_type=df_results_uid['best_model'].iloc[0]
    prices = np.linspace(df_uid[price_col].min(), df_uid[price_col].max(), 100)


    if model_type == "power":
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Power Demand Curve'
    elif model_type == "exponential":
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Exponential Demand Curve'
    elif model_type == "linear":
        quantities = linear_demand_equation(prices, a_linear, b_linear)
        label = 'Linear Demand Curve'
    else: 
        print('typo in model type, power, exponential or linear')

    plt.plot(quantities, prices, color="blue", label='Quicklizard Model')
    plt.scatter(df_uid['units'],
                df_uid[price_col],
                s=df_uid['days']*10, marker='+', 
                color="red", 
                label='Actual Data (Average units sold, size is proportional to the number of days at this price)')

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title + label)
    plt.grid(True)
        # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()

def plot_model_type(a_linear, b_linear, model_type, prices = np.linspace(50, 100, 100), title='')-> None:
    """plot_demand_curves."""

    if model_type == "power":
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Power Demand Curve'
    if model_type == "exponential":
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Exponential Demand Curve'
    if model_type == "linear":
        quantities = linear_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Linear Demand Curve'
    else: 
        print('typo in model type, power, exponential or linear')

    plt.plot(quantities, prices, label=label, color="blue")

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title + label)
    plt.grid(True)

    plt.show()

def plot_demand_curves(a_linear=10, b_linear = -0.1,
                       a_exponential=30, b_exponential=-0.03,
                       a_power=1000000, b_power=-3,
                       prices = np.linspace(50, 100, 100)
                       ) -> None:
    """plot_demand_curves."""

    # Calculate quantity demanded for each price
    quantities_linear = linear_demand_equation(prices, a_linear, b_linear)
    quantities_exponential = exponential_demand_equation(
        prices, a_exponential, b_exponential
    )
    quantities_power = power_demand_equation(prices, a_power, b_power)

    # Plot demand curves with different colors
    plt.plot(
        quantities_linear, prices, label="Linear Demand - Q = 10 + -0.1xP", color="blue"
    )
    plt.plot(
        quantities_exponential,
        prices,
        color="red",
        label="Exponential Demand - Q = 30 * exp(-0.03xP) - Elasticity = -0.03*P",
    )
    plt.plot(
        quantities_power,
        prices,
        color="green",
        label="Power Demand - Q = 1000000 * P^(-3) - Constant Elasticity = -3",
    )

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title("Demand Curves")
    plt.grid(True)

    # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()


def plot_model_and_prices_buffer(df_results, df_data, uid, title='') -> BytesIO:
    """
    Plot the model and prices.

    Parameters:
        df_results (pd.DataFrame): DataFrame containing model results.
        df_data (pd.DataFrame): DataFrame containing data.
        uid (int): Unique identifier.
        title (str): Title of the plot (optional).

    Returns:
        BytesIO: Buffer containing the plot.
    """
    df_results_uid = df_results[df_results.uid==uid]
    df_uid = df_data[df_data.uid==uid]
    a_linear=df_results_uid['best_model_a'].iloc[0]
    b_linear=df_results_uid['best_model_b'].iloc[0]
    model_type=df_results_uid['best_model'].iloc[0]
    prices = np.linspace(df_uid['round_price'].min(), df_uid['round_price'].max(), 100)

    fig, ax = plt.subplots()

    if model_type == "power":
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Power Demand Curve'
    elif model_type == "exponential":
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Exponential Demand Curve'
    elif model_type == "linear":
        quantities = linear_demand_equation(prices, a_linear, b_linear)
        label = 'Linear Demand Curve'
    else:
        print('typo in model type, power, exponential or linear')
        return None

    ax.plot(quantities, prices, color="blue", label='Quicklizard Model')
    ax.scatter(df_uid['units'], df_uid['round_price'], s=df_uid['days']*10, marker='+', color="red", label='Actual Data (Average units sold, size is proportional to the number of days at this price)')

    ax.set_xlabel("Quantity")
    ax.set_ylabel("Price")
    ax.set_title(title + label + f" - UID: {uid} - Elasticity: {df_results_uid['best_model_elasticity'].iloc[0]:.2f}")
    ax.grid(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')

    plt.close(fig)  # Close the figure to free memory

    buffer.seek(0)
    return buffer



def plot_model_and_prices(df_results, df_data, uid, title='')-> None:
    print('uid:', uid)
    df_results_uid = df_results[df_results.uid==uid]
    print('best_model_elasticity:', df_results_uid['best_model_elasticity'].iloc[0])
    df_uid = df_data[df_data.uid==uid]
    a_linear=df_results_uid['best_model_a'].iloc[0]
    b_linear=df_results_uid['best_model_b'].iloc[0]
    model_type=df_results_uid['best_model'].iloc[0]
    prices = np.linspace(df_uid['round_price'].min(), df_uid['round_price'].max(), 100)


    if model_type == "power":
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Power Demand Curve'
    elif model_type == "exponential":
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = 'Exponential Demand Curve'
    elif model_type == "linear":
        quantities = linear_demand_equation(prices, a_linear, b_linear)
        label = 'Linear Demand Curve'
    else: 
        print('typo in model type, power, exponential or linear')

    plt.plot(quantities, prices, color="blue", label='Quicklizard Model')
    plt.scatter(df_uid['units'], df_uid['round_price'], s=df_uid['days']*10, marker='+', color="red", label='Actual Data (Average units sold, size is proportional to the number of days at this price)')

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title + label)
    plt.grid(True)
        # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()

# Example usage:
# plot_demand_curves()
