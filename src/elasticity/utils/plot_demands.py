"""Module of plot_demands."""

import logging
import multiprocessing
import traceback
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elasticity.model.model import estimate_coefficients
from elasticity.utils.consts import EXPONENTIAL, LINEAR, POWER
from ql_toolkit.s3 import io as s3io

plt.switch_backend("agg")


def linear_demand_equation(price: float, a: float, b: float) -> float:
    """Linear demand model: Q = a - bP."""
    return a + b * price


def exponential_demand_equation(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP)."""
    return a * np.exp(b * price)


def power_demand_equation(price: float, a: float, b: float) -> float:
    """Power demand model: Q = a * P^(-b)."""
    return a * (price ** (b))


def generate_data(
    price_points: int = 5,
    start_price: float = 20,
    stop_price: float = 50,
    model_type: str = "linear",
    a: float = 300,
    b: float = -2,
    error: float = 0,
    random_state: int = 10,
) -> pd.DataFrame:
    """Generate synthetic data for price and quantity.

    Parameters:
    - price_points (int): Number of price points.
    - start_price (float): Starting price.
    - stop_price (float): Ending price.
    - model_type (str): Type of demand model
    ("linear", "power", or "exponential").
    - a (float): Parameter 'a' for demand model equations.
    - b (float): Parameter 'b' for demand model equations.
    - error (float): Standard deviation of the error
    term to be added to quantity.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - DataFrame: Pandas DataFrame containing generated data
    with columns 'price', 'quantity', and 'days'.
    """
    np.random.seed(random_state)

    price_range = np.linspace(start=start_price, stop=stop_price, num=price_points)

    # Generate random integers for days following a normal distribution
    weights = np.random.normal(loc=3, scale=1, size=len(price_range)).astype(int)

    if model_type == "linear":
        quantity = linear_demand_equation(price_range, a, b)
    elif model_type == "power":
        quantity = power_demand_equation(price_range, a, b)
    elif model_type == "exponential":
        quantity = exponential_demand_equation(price_range, a, b)

    # add error to quantity relative to the weights
    if error != 0:
        max_weight = np.max(weights)
        scaled_weights = weights / max_weight
        error_factor = np.random.normal(loc=1, scale=error, size=len(quantity))
        quantity *= error_factor * scaled_weights

    # Clip quantity to be 0.001 to avoid negative values
    logging.info(f"before clip {quantity}")
    quantity = np.clip(quantity, 0.001, None)
    logging.info(f"after clip {quantity}")
    return pd.DataFrame({"price": price_range, "quantity": quantity, "days": weights})


def plot_quantity_and_prices_from_df(
    df: pd.DataFrame,
    title: str = "",
    quantity_col: str = "quantity",
    price_col: str = "price",
    days_col: str = "days",
    outlier_col: str = "outlier_quantity",
) -> None:
    """Plot quantity and prices from the DataFrame with outlier differentiation.

    This function creates a scatter plot of quantities and prices from the provided DataFrame.
    It differentiates between actual data and outlier data based on a boolean mask column.
    The size of the markers in the scatter plot is proportional to the number of days
    at each price point.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to plot.
        title (str, optional): The title of the plot. Defaults to an empty string.
        quantity_col (str, optional): The column name for quantities in the DataFrame.
        Defaults to "quantity".
        price_col (str, optional): The column name for prices in the DataFrame. Defaults to "price".
        days_col (str, optional): The column name for the number of days in the DataFrame.
        Defaults to "days".
        outlier_col (str, optional): The column name for the outlier boolean mask in the DataFrame.
        Defaults to "outlier_quantity".

    Returns:
        None: This function does not return any value. It only creates a plot.

    """
    outlier_mask = df[outlier_col]  # Boolean mask for outliers
    plt.scatter(
        df[quantity_col][~outlier_mask],  # Plot non-outlier points
        df[price_col][~outlier_mask],
        s=df[days_col][~outlier_mask] * 10,
        marker="+",
        color="blue",
        label=(
            "Actual Data (Avg units sold, size is prop. to "
            "the nb. of days at this price"
        ),
    )
    plt.scatter(
        df[quantity_col][outlier_mask],  # Plot outlier points
        df[price_col][outlier_mask],
        s=df[days_col][outlier_mask] * 10,
        marker="+",
        color="red",  # Change color for outliers
        label="Outlier Data",
    )

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title)
    plt.grid(True)
    # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()


def plot_model_and_prices_from_df(
    df: pd.DataFrame,
    model_type: str = LINEAR,
    title: str = "",
    quantity_col: str = "quantity",
    price_col: str = "price",
    days_col: str = "days",
    outlier_col: str = "outlier_quantity",
) -> None:
    """Plot demand curves from model results and actual data.

    This function estimates the coefficients for a specified demand model type
    (linear, exponential, or power) using the provided DataFrame and plots the
    resulting demand curve alongside the actual data points. Outliers are differentiated
    in the scatter plot.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to plot.
        model_type (str, optional): The type of demand model to fit. Defaults to LINEAR.
        title (str, optional): The title of the plot. Defaults to an empty string.
        quantity_col (str, optional): The column name for quantities in the DataFrame.
        Defaults to "quantity".
        price_col (str, optional): The column name for prices in the DataFrame.
        Defaults to "price".
        days_col (str, optional): The column name for the number of days in the DataFrame.
        Defaults to "days".
        outlier_col (str, optional): The column name for the outlier boolean mask in the DataFrame.
        Defaults to "outlier_quantity".

    Returns:
        None: This function does not return any value. It only creates a plot.

    Raises:
        ValueError: If an invalid model type is provided.
    """
    estimation_result = estimate_coefficients(
        data=df,
        model_type=model_type,
        price_col=price_col,
        quantity_col=quantity_col,
        weights_col=days_col,
    )

    prices = df[price_col]

    if model_type == POWER:
        quantities = power_demand_equation(
            prices, np.exp(estimation_result.aic), estimation_result.b
        )
        label = "Power Demand Curve"
    elif model_type == EXPONENTIAL:
        quantities = exponential_demand_equation(
            prices, np.exp(estimation_result.a), estimation_result.b
        )
        label = "Exponential Demand Curve"
    elif model_type == LINEAR:
        quantities = linear_demand_equation(
            prices, float(estimation_result.a), float(estimation_result.b)
        )
        label = "Linear Demand Curve"
    else:
        logging.info("typo in model type, power, exponential or linear")

    outlier_mask = df[outlier_col]  # Boolean mask for outliers

    plt.plot(quantities, prices, color="blue", label="Quicklizard Model")
    plt.scatter(
        df[quantity_col][~outlier_mask],  # Plot non-outlier points
        df[price_col][~outlier_mask],
        s=df[days_col][~outlier_mask] * 10,
        marker="+",
        color="blue",
        label=(
            "Actual Data (Avg units sold, size is prop. to "
            "the nb. of days at this price"
        ),
    )
    plt.scatter(
        df[quantity_col][outlier_mask],  # Plot outlier points
        df[price_col][outlier_mask],
        s=df[days_col][outlier_mask] * 10,
        marker="+",
        color="red",  # Change color for outliers
        label="Outlier Data",
    )

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(
        title
        + label
        + "= elasticity: "
        + str(estimation_result.elasticity)
        + " +/- "
        + str(round(estimation_result.elasticity_error_propagation, 2))
    )
    plt.grid(True)
    # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()


def plot_model_and_prices(
    df_results: pd.DataFrame,
    df_data: pd.DataFrame,
    uid: str,
    title: str = "",
    quantity_col: str = "quantity",
    price_col: str = "price",
    days_col: str = "days",
    outlier_col: str = "outlier_quantity",
) -> None:
    """Plot demand curves from model results and actual data for a specific UID.

    This function extracts model parameters from a results DataFrame and actual data
    for a specific UID from a data DataFrame. It then plots the resulting demand curve
    alongside the actual data points. Outliers are differentiated in the scatter plot.

    Args:
        df_results (pd.DataFrame): The DataFrame containing the model results.
        df_data (pd.DataFrame): The DataFrame containing the actual data.
        uid (str): The unique identifier for the data to be plotted.
        title (str, optional): The title of the plot. Defaults to an empty string.
        quantity_col (str, optional): The column name for quantities in the DataFrame.
        Defaults to "quantity".
        price_col (str, optional): The column name for prices in the DataFrame.
        Defaults to "price".
        days_col (str, optional): The column name for the number of days in the DataFrame.
        Defaults to "days".
        outlier_col (str, optional): The column name for the outlier boolean mask in the DataFrame.
        Defaults to "outlier_quantity".

    Returns:
        None: This function does not return any value. It only creates a plot.
    """
    df_results_uid = df_results[df_results.uid == uid]
    df_uid = df_data[df_data.uid == uid]
    a_linear = df_results_uid["best_a"].iloc[0]
    b_linear = df_results_uid["best_b"].iloc[0]
    model_type = df_results_uid["best_model"].iloc[0]
    elasticity = df_results_uid["best_elasticity"].iloc[0]
    error = df_results_uid["best_elasticity_error_propagation"].iloc[0]
    prices = np.linspace(df_uid[price_col].min(), df_uid[price_col].max(), 100)

    if model_type == POWER:
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Power Demand Curve"
    elif model_type == EXPONENTIAL:
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Exponential Demand Curve"
    elif model_type == LINEAR:
        quantities = linear_demand_equation(prices, a_linear, b_linear)
        label = "Linear Demand Curve"
    else:
        logging.info("typo in model type, power, exponential or linear")

    outlier_mask = df_uid[outlier_col]  # Boolean mask for outliers

    plt.plot(quantities, prices, color="blue", label="Quicklizard Model")
    plt.scatter(
        df_uid[quantity_col][~outlier_mask],  # Plot non-outlier points
        df_uid[price_col][~outlier_mask],
        s=df_uid[days_col][~outlier_mask] * 10,
        marker="+",
        color="green",
        label=(
            "Actual Data (Avg units sold, size is prop. to "
            "the nb. of days at this price"
        ),
    )
    plt.scatter(
        df_uid[quantity_col][outlier_mask],  # Plot outlier points
        df_uid[price_col][outlier_mask],
        s=df_uid[days_col][outlier_mask] * 10,
        marker="+",
        color="red",  # Change color for outliers
        label="Outlier Data",
    )

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(
        title
        + label
        + "= elasticity: "
        + str(elasticity)
        + " +/- "
        + str(round(error, 2))
    )
    plt.grid(True)
    # Place legend outside the plot at the bottom
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    plt.show()


def plot_model_type(
    a_linear: float,
    b_linear: float,
    model_type: str,
    prices: np.ndarray | None = None,
    title: str = "",
) -> None:
    """plot_demand_curves."""
    if prices is None:
        prices = np.ndarray = np.linspace(50, 100, 100)
    if model_type == "power":
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Power Demand Curve"
    if model_type == "exponential":
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Exponential Demand Curve"
    if model_type == "linear":
        quantities = linear_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Linear Demand Curve"
    else:
        logging.info("typo in model type, power, exponential or linear")

    plt.plot(quantities, prices, label=label, color="blue")

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(title + label)
    plt.grid(True)

    plt.show()


def plot_demand_curves(
    a_linear: float = 10,
    b_linear: float = -0.1,
    a_exponential: float = 30,
    b_exponential: float = -0.03,
    a_power: float = 1000000,
    b_power: float = -3,
    prices: np.ndarray | None = None,
) -> None:
    """plot_demand_curves."""
    if prices is None:
        prices = np.ndarray = np.linspace(50, 100, 100)
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


def plot_model_and_prices_buffer(
    df_results: pd.DataFrame, df_data: pd.DataFrame, uid: str, title: str = ""
) -> BytesIO:
    """Plot the model and prices.

    Parameters:
        df_results (pd.DataFrame): DataFrame containing model results.
        df_data (pd.DataFrame): DataFrame containing data.
        uid (int): Unique identifier.
        title (str): Title of the plot (optional).

    Returns:
        BytesIO: Buffer containing the plot.
    """
    df_results_uid = df_results[df_results.uid == uid]
    df_uid = df_data[df_data.uid == uid]
    a_linear = df_results_uid["best_a"].iloc[0]
    b_linear = df_results_uid["best_b"].iloc[0]
    model_type = df_results_uid["best_model"].iloc[0]
    elasticity = df_results_uid["best_elasticity"].iloc[0]
    error = df_results_uid["best_elasticity_error_propagation"].iloc[0]
    prices = np.linspace(df_uid["round_price"].min(), df_uid["round_price"].max(), 100)

    fig, ax = plt.subplots()

    if model_type == "power":
        quantities = power_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Power Demand Curve"
    elif model_type == "exponential":
        quantities = exponential_demand_equation(prices, np.exp(a_linear), b_linear)
        label = "Exponential Demand Curve"
    elif model_type == "linear":
        quantities = linear_demand_equation(prices, a_linear, b_linear)
        label = "Linear Demand Curve"
    else:
        logging.info("typo in model type, power, exponential or linear")
        return None

    ax.plot(quantities, prices, color="blue", label="Quicklizard Model")
    ax.scatter(
        df_uid["units"],
        df_uid["round_price"],
        s=df_uid["days"] * 10,
        marker="+",
        color="red",
        label="Actual Data(Avg units sold, size is prop. to the nb. of days at this price)",
    )

    plt.plot(quantities, prices, color="blue", label="Quicklizard Model")
    plt.scatter(
        df_uid["units"],
        df_uid["round_price"],
        s=df_uid["days"] * 10,
        marker="+",
        color="red",
        label="Actual Data(Avg units sold, size is prop. to the nb. of days at this price)",
    )

    plt.xlabel("Quantity")
    plt.ylabel("Price")
    plt.title(
        title
        + label
        + "= elasticity: "
        + str(elasticity)
        + " +/- "
        + str(round(error, 2))
    )
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fancybox=True)

    # Save the plot to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(fig)  # Close the figure to free memory

    buffer.seek(0)
    return buffer


def save_graph(
    uid: pd.DataFrame,
    df_results: pd.DataFrame,
    df_by_price: pd.DataFrame,
    client_key: str,
    channel: str,
    end_date: str,
) -> None:
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
        s3io.upload_to_s3(
            s3_dir=f"data_science/eval_results/elasticity/graphs/{client_key}/{channel}/",
            file_name=f"{uid}_{end_date}.png",
            file_obj=buffer,
        )
    except Exception:
        # Log the exception traceback
        traceback.print_exc()


def run_save_graph_top10(
    df_results: pd.DataFrame,
    df_by_price: pd.DataFrame,
    client_key: str,
    channel: str,
    end_date: str,
) -> None:
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
    elasticity_uids_top10 = (
        df_results[df_results["quality_test"]]
        .sort_values("best_relative_absolute_error")[:10]["uid"]
        .unique()
    )

    for uid in elasticity_uids_top10:
        save_graph(uid, df_results, df_by_price, client_key, channel, end_date)


def run_save_graph_parallel(
    df_results: pd.DataFrame,
    df_by_price: pd.DataFrame,
    client_key: str,
    channel: str,
    end_date: str,
) -> None:
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
    elasticity_uids = (
        df_results[df_results["quality_test"]]
        .sort_values("best_relative_absolute_error")["uid"]
        .unique()
    )

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
