"""Module of utils."""

import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from elasticity.data.configurator import DataColumns


def calculate_last_values(
    input_df: pd.DataFrame, data_columns: DataColumns
) -> pd.DataFrame:
    """Get last price and date for each uid in a new df.

    Args:
        input_df (pd.DataFrame): The input DataFrame.
        data_columns (DataColumns): An instance of the DataColumns class.

    Returns:
        pd.DataFrame: A new DataFrame with columns uid, last_price, and last_date.
    """
    # Group by uid and get the last entry for each group
    last_values_df = (
        input_df[[data_columns.uid, data_columns.date]]
        .groupby(data_columns.uid)
        .agg({data_columns.date: "max"})
        .reset_index()
    )

    # Merge back to get the last price corresponding to the last date for each uid
    last_values_df = last_values_df.merge(
        input_df[[data_columns.uid, data_columns.date, data_columns.price]],
        on=[data_columns.uid, data_columns.date],
        how="left",
    )

    return last_values_df.rename(
        columns={data_columns.price: "last_price", data_columns.date: "last_date"}
    )


def get_revenue(
    df_revenue: pd.DataFrame, uid_col: str, total_uid: int
) -> Tuple[int, float, pd.DataFrame]:
    """Calculate the revenue and revenue percentage by uid.

    Args:
        df_revenue (pandas.DataFrame): The input dataframe containing the revenue data.
        uid_col (str): The name of the column representing the unique identifier.
        total_uid (int): The total number of unique identifiers.

    Returns:
        tuple: A tuple containing:
            - total_uid (Optional[int]): The total number of unique identifiers.
            - total_revenue (Optional[float]): The total revenue.
            - df_revenue_uid (Optional[pd.DataFrame]): A dataframe with the revenue and
            revenue percentage for each unique identifier.
    """
    total_uid = df_revenue[uid_col].nunique()

    df_revenue["revenue"] = df_revenue["shelf_price"] * df_revenue["units"]
    total_revenue = df_revenue["revenue"].sum()

    df_revenue_uid = df_revenue.groupby(uid_col)["revenue"].sum().reset_index()
    df_revenue_uid["revenue_percentage"] = df_revenue_uid["revenue"] / total_revenue

    logging.info(f"Total uid: {total_uid}")
    return total_uid, total_revenue, df_revenue_uid


def round_down_to_nearest_half(number: float) -> float:
    """Rounds down a given number to the nearest half.

    Args:
        number (float): The number to be rounded down.

    Returns:
        float: The rounded down number.

    Example:
        >>> round_down_to_nearest_half(3.7)
        3.5
        >>> round_down_to_nearest_half(6.2)
        6.0
        >>> round_down_to_nearest_half(0.1)
        0.0
    """
    nearest_integer = int(number)
    decimal_part = number - nearest_integer
    nearest_half = 0 if decimal_part < 0.5 else 0.5
    return nearest_integer + nearest_half


def round_price_effect(price: float) -> float:
    """Rounds the price by taking into consideration the price effect.

    - For prices less than 10, rounds down to the nearest half at increments of 0.05.
    - For prices less than 50, rounds down to the nearest half at increments of 0.5.
    - For prices between 50 and 500, takes the floor of the whole number.
    - For prices greater than 500, rounds down to the nearest half at increments of 5.

    Example:
        >>> round_price_effect(9.54)
        9.5
        >>> round_price_effect(9.99)
        9.95
        >>> round_price_effect(15.82)
        15.5
        >>> round_price_effect(103.5)
        103
        >>> round_price_effect(687)
        685
    """
    if np.isnan(price) or (price < 0):
        return np.nan
    if price < 10:
        return round_down_to_nearest_half(price * 10) / 10
    if price < 50:
        return round_down_to_nearest_half(price)
    if price > 500:
        return int(round_down_to_nearest_half(price / 10) * 10)
    return int(price)


def preprocess_by_price(
    input_df: pd.DataFrame,
    data_columns: Optional[DataColumns] = None,
) -> pd.DataFrame:
    """Grouping data by price and calculate weight representing the number of days.

    Args:
        input_df (pd.DataFrame): The input DataFrame containing the data to be preprocessed.
        data_columns (DataColumns, optional): An instance of the DataColumns class that specifies
        the column names. Defaults to an empty instance of DataColumns.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    if data_columns is None:
        data_columns = DataColumns()
    uid_col = data_columns.uid
    price_col = data_columns.price
    quantity_col = data_columns.quantity
    date_col = data_columns.date

    df_by_price_norm = (
        input_df.groupby([uid_col, price_col, "outlier_quantity"])
        .agg({quantity_col: "mean", date_col: "count"})
        .reset_index()
        .sort_values(by=[uid_col, price_col])
        .rename(columns={date_col: data_columns.weight})
    )

    df_by_price_norm[quantity_col] = df_by_price_norm[quantity_col] + 0.001

    return df_by_price_norm


def uid_with_price_changes(
    input_df: pd.DataFrame,
    price_changes: int = 5,
    threshold: float = 0.01,
    data_columns: Optional[DataColumns] = None,
) -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria.

    DataFrame of uids with more than 'price_changes' price changes
    that are more than 'threshold' on non outlier quantity

    Args:
        input_df (pd.DataFrame): The input DataFrame containing the data.
        price_changes (int, optional): The minimum number of price changes required for a user ID
        to be considered an elasticity candidate. Defaults to 4.
        threshold (float, optional): The threshold value for significant price changes.
        Defaults to 0.01.
        data_columns (DataColumns, optional): An instance of the DataColumns class containing the
        column names used in the DataFrame. Defaults to DataColumns().

    Returns:
        pd.DataFrame: A DataFrame containing the user IDs with significant price changes.


    """
    if data_columns is None:
        data_columns = DataColumns()
    uid_col = data_columns.uid
    price_col = data_columns.price

    # Preprocess data
    df_by_price = preprocess_by_price(input_df=input_df, data_columns=data_columns)

    # TODO: VALIDATE OR DELETE THIS
    # take out price with no sells
    df_by_price = df_by_price[df_by_price[data_columns.quantity] > 0.001]

    # Calculate price change percentage.
    # df_by_price already sorted in preprocess_by_price
    df_by_price["price_change"] = df_by_price.groupby([uid_col, "outlier_quantity"])[
        price_col
    ].pct_change()
    df_by_price["price_change_without_outliers"] = np.where(
        df_by_price["outlier_quantity"], 0, df_by_price["price_change"]
    )

    # Filter user IDs with significant price changes
    return (
        df_by_price[df_by_price["price_change_without_outliers"].abs() > threshold]
        .groupby(uid_col)[price_col]
        .nunique()
        .loc[lambda x: x > price_changes]
        .index.tolist()
    )


def outliers_iqr_filtered(
    ys: list[float],
    filter_threshold: float = 0.001,
    outlier_threshold: float = 10,
    range_threshold: float = 15,
    q: float = 1.5,
    quartile: float = 15,
) -> list[bool]:
    """Detect outliers in a list of values using the Interquartile Range (IQR) method.

    Args:
        ys (list[float]): The list of values to analyze.
        filter_threshold (float, optional): The threshold to eliminate no conversions day
        outlier_threshold (float, optional): The threshold above which values are considered
            potential outliers.
            Defaults to 10.
        range_threshold (float, optional): The threshold for the range of filtered values.
            If the range is less than or equal to this value, no outliers are detected.
            Defaults to 15.
        q (float, optional): The coefficient used to determine the bounds for outliers.
            Defaults to 1.5.
        quartile (float, optional): The percentile used to calculate the quartiles.
            Defaults to 15.

    Returns:
        list[bool]: A boolean list indicating whether each value is
        an outlier (True) or not (False).
    """
    ys = np.array(ys)
    filtered_ys = ys[ys > filter_threshold]

    if len(filtered_ys) == 0:
        return [False] * len(ys)

    filtered_range = np.max(filtered_ys) - np.min(filtered_ys)

    if filtered_range <= range_threshold:
        return [False] * len(ys)

    quartile_1, quartile_3 = np.percentile(filtered_ys, [quartile, 100 - quartile])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - q * iqr
    upper_bound = quartile_3 + q * iqr

    outliers = (ys < lower_bound) | (ys > upper_bound)
    # Set values less than or equal to the outlier_threshold to False
    outliers[ys <= outlier_threshold] = False

    return outliers


def uid_with_min_conversions(
    input_df: pd.DataFrame,
    min_conversions_days: int = 10,
    uid_col: str = "uid",
    quantity_col: str = "units",
) -> pd.DataFrame:
    """Filter the DataFrame to get uid with a least 'min_conversions_days' days with conversions.

    Args:
        input_df (pd.DataFrame): The input DataFrame containing the data.
        min_conversions_days (int, optional): The minimum number of conversion days required.
        Defaults to 10.
        uid_col (str, optional): The column name for the unique identifier. Defaults to "uid".
        quantity_col (str, optional): The column name for the quantity. Defaults to "units".

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    return (
        input_df[input_df[quantity_col] > 0]
        .groupby(uid_col)
        .filter(lambda x: len(x) >= min_conversions_days)[uid_col]
        .unique()
    )


def initialize_dates(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, str]:
    """Initializes the start and end dates for data processing.

    Args:
        start_date (str, optional): The start date in the format 'YYYY-MM-DD'.
        Defaults to None.
        end_date (str, optional): The end date in the format 'YYYY-MM-DD'.
        Defaults to None.

    Returns:
        tuple: A tuple containing the initialized start date and end date.

    If either start_date or end_date is None, the function calculates the default
    values based on the current date.
    The default end_date is two days ago, with the day set to the first day of
    the previous month.
    The default start_date is 11 months before the end_date.

    Example:
        start_date, end_date = initialize_dates()
        print(start_date, end_date)
        # Output: '2021-01-01', '2021-12-01'
    """
    if end_date is None:
        end_date = datetime.now().replace(day=1) - relativedelta(days=1)
        end_date = str(end_date.strftime("%Y-%m-%d"))

    if start_date is None:
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(months=11)
        start_date = str(start_date_dt.strftime("%Y-%m-%d"))

    return start_date, end_date


# TODO: REFACTOR
# Not in use now
# def get_elasticity_candidates(
#     input_df: pd.DataFrame,
#     nb_months: int = 3,
#     min_conversion_day_per_month: int = 10,
#     price_changes: int = 4,
#     threshold: float = 0.01,
#     uid_col: str = "uid",
#     date_col: str = "date",
#     price_col: str = "price_from_revenue",
#     quantity_col: str = "units",
# ) -> pd.DataFrame:
#     """Filter the DataFrame to obtain elasticity candidates based on specified criteria.

#     Args:
#         input_df (DataFrame): Input DataFrame containing user interactions.
#         nb_months (int): Number of months to consider for conversion.
#         min_conversion_day_per_month (int): Minimum conversion days per month.
#         price_changes (int): Minimum number of price changes.
#         threshold (float): Minimum percentage change to consider significant.
#         uid_col (str): Column name for user IDs.
#         date_col (str): Column name for dates.
#         price_col (str): Column name for prices.
#         quantity_col (str): Column name for quantity.

#     Returns:
#         DataFrame: Filtered DataFrame containing elasticity candidates.
#     """
#     logging.info(f"Number of rows: {input_df.shape[0]}")
#     logging.info(f"Number of unique user IDs: {input_df[uid_col].nunique()}")
#     # Get user IDs with minimum conversion days
#     min_conversion_day = nb_months * min_conversion_day_per_month
#     uid_min_conversion_day = (
#         input_df[input_df[quantity_col] > 0]
#         .groupby(uid_col)
#         .filter(lambda x: len(x) >= min_conversion_day)[uid_col]
#         .unique()
#     )
#     input_df = input_df[input_df[uid_col].isin(uid_min_conversion_day)]
#     logging.info(
#         "Number of user IDs with more than",
#         min_conversion_day,
#         "days with conversion:",
#         min_conversion_day,
#         len(uid_min_conversion_day),
#     )
#     # Preprocess data
#     df_norm = preprocess_by_price(
#         input_df=input_df,
#         uid_col=uid_col,
#         date_col=date_col,
#         price_col=price_col,
#         quantity_col=quantity_col,
#     )
#     # Calculate price change percentage
#     df_norm = df_norm.sort_values(by=[uid_col, price_col])
#     df_norm["price_change"] = df_norm.groupby(uid_col)[price_col].pct_change()

#     # Filter user IDs with significant price changes
#     uid_price_changes = (
#         df_norm[df_norm["price_change"].abs() > threshold]
#         .groupby(uid_col)[price_col]
#         .nunique()
#         .loc[lambda x: x > price_changes]
#         .index.tolist()
#     )
#     logging.info(
#         "Number of user IDs with price changes of more than",
#         threshold * 100,
#         "%:",
#         len(uid_price_changes),
#     )
#     df_gold_candidates = input_df[input_df[uid_col].isin(uid_price_changes)]
#     logging.info(f"Number of gold candidates: {df_gold_candidates[uid_col].nunique()}")
#     return df_gold_candidates.sort_values(by=[uid_col, date_col])
