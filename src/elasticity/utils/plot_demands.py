"""Module of plot_demands."""

import multiprocessing
from io import BytesIO
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ql_toolkit.s3 import io as s3io


def linear_demand_equation(price: float, a: float, b: float) -> float:
    """Linear demand model: Q = a - bP."""
    return a + b * price

def exponential_demand_equation(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP)."""
    return a * np.exp(b * price)

def power_demand_equation(price: float, a: float, b: float) -> float:
    """Power demand model: Q = a * P^(-b)."""
    return a * (price ** (b))

def plot_model_and_prices(df_results: pd.DataFrame,
                          df_data: pd.DataFrame,
                          uid: str,
                          title: str='',
                          price_col: str='round_price')-> None:
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
    plt.scatter(
        df_uid['units'],
        df_uid[price_col],
        s=df_uid['days']*10,
        marker='+',
        color="red",
        label=('Actual Data (Avg units sold, size is prop. to '
               'the nb. of days at this price'))

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title + label)
    plt.grid(True)
        # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()

def plot_model_type(a_linear: float,
                    b_linear: float,
                    model_type: str,
                    prices: np.ndarray = np.linspace(50, 100, 100),
                    title: str ='')-> None:
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

def plot_demand_curves(a_linear: float=10, b_linear: float = -0.1,
                       a_exponential: float=30, b_exponential: float=-0.03,
                       a_power: float=1000000, b_power:float=-3,
                       prices: np.ndarray = np.linspace(50, 100, 100)
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


def plot_model_and_prices_buffer(df_results: pd.DataFrame,
                                 df_data: pd.DataFrame,
                                 uid: str,
                                 title: str='') -> BytesIO:
    """Plot the model and prices.

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
    ax.scatter(df_uid['units'],
               df_uid['round_price'],
               s=df_uid['days']*10,
               marker='+',
               color="red",
               label='Actual Data(Avg units sold, size is prop. to the nb. of days at this price)')

    plt.plot(quantities, prices, color="blue", label='Quicklizard Model')
    plt.scatter(df_uid['units'],
                df_uid['round_price'],
                s=df_uid['days']*10,
                marker='+',
                color="red",
                label='Actual Data(Avg units sold, size is prop. to the nb. of days at this price)')

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title + label)
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig)  # Close the figure to free memory

    buffer.seek(0)
    return buffer


def save_graph(uid: pd.DataFrame,
               df_results: pd.DataFrame,
               df_by_price: pd.DataFrame,
               client_key: str,
               channel: str,
               end_date: str) -> None:
    """Save graph for a given UID to S3.

    Args:
        uid (Any): Unique identifier for the graph.
        df_results (Any): DataFrame containing results.
        df_by_price (Any): DataFrame containing prices.
        client_key (str): Key identifying the client.
        channel (str): Channel identifier.
        end_date (str): End date for the graph.

    Returns:
        None
    """
    try:
        buffer = plot_model_and_prices_buffer(df_results, df_by_price, uid)
        s3io.upload_to_s3(s3_dir=f"data_science/eval_results/elasticity/graphs/{client_key}/{channel}/",
                          file_name=f"{uid}_{end_date}.png",
                          file_obj=buffer)
    except Exception:
        # Log the exception traceback
        traceback.print_exc()
    return None

def run_save_graph_top10(df_results: pd.DataFrame,
                            df_by_price: pd.DataFrame,
                            client_key: str,
                            channel: str,
                            end_date: str) -> None:
    """Run save_graph function in parallel for UIDs based on sorting criteria.

    Args:
        df_results (Any): DataFrame containing results.
        df_by_price (Any): DataFrame containing prices.
        client_key (str): Key identifying the client.
        channel (str): Channel identifier.
        end_date (str): End date for the graph.

    Returns:
        None
    """
    elasticity_uids_top10 = df_results[df_results["quality_test"]].sort_values(
        'best_mean_relative_error')[:10]['uid'].unique()


    for uid in elasticity_uids_top10:
        save_graph(uid, df_results, df_by_price, client_key, channel, end_date)

def run_save_graph_parallel(df_results: pd.DataFrame,
                            df_by_price: pd.DataFrame,
                            client_key: str,
                            channel: str,
                            end_date: str) -> None:
    """Run save_graph function in parallel for UIDs based on sorting criteria.

    Args:
        df_results (Any): DataFrame containing results.
        df_by_price (Any): DataFrame containing prices.
        client_key (str): Key identifying the client.
        channel (str): Channel identifier.
        end_date (str): End date for the graph.

    Returns:
        None
    """
    elasticity_uids = df_results[df_results["quality_test"]].sort_values(
        'best_mean_relative_error')['uid'].unique()


    # Apply the function in parallel for each UID
    pool = multiprocessing.Pool()  # Use the default number of processes
    _ = pool.starmap(
        save_graph,
        [
            (uid, df_results, df_by_price, client_key, channel, end_date)
            for uid in elasticity_uids
        ],
    )
    pool.close()
    pool.join()
