"""Module of utils."""
import logging

import numpy as np
import pandas as pd


def round_down_to_nearest_half(number: float) -> float:
    """Rounds down to the nearest half."""
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
    if np.isnan(price):
        return np.nan
    if price < 10:
        return round_down_to_nearest_half(price * 10) / 10
    if price < 50:
        return round_down_to_nearest_half(price)
    if price > 500:
        return int(round_down_to_nearest_half(price / 10) * 10)
    return int(price)


def preprocess_by_day(input_df: pd.DataFrame,
                      uid_col: str = 'uid',
                      date_col: str = 'date',
                      price_col: str = 'price',
                      quantity_col: str = 'units',
                      ) ->tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the DataFrame by grouping data by day.

    Parameters:
    - df (DataFrame): DataFrame containing the data.
    - uid_col (str): Column name for the UID.
    - date_col (str): Column name for the date.
    - price_col (str): Column name for the price data.
    - quantity_col (str): Column name for the quantity data.
    - median_price_col (str): Column name for the median price.
    - median_quantity_col (str): Column name for the median quantity.

    Returns:
    - df_by_day (DataFrame): DataFrame grouped by day.
    - df_by_price_norm (DataFrame): DataFrame grouped by price normalization.
    """
    input_df.loc[:, date_col] = pd.to_datetime(input_df[date_col])
    input_df.loc[:, price_col] = round(input_df[price_col], 1)

    # Create a pivot table to reshape the data
    pivot_df = input_df.pivot_table(index=uid_col,
                                    columns=date_col,
                                    values=quantity_col,
                                    fill_value=0)

    # Reindex the pivot table to fill in missing dates with 0s
    pivot_df = pivot_df.reindex(
        columns=pd.date_range(start=input_df[date_col].min(), end=input_df[date_col].max()),
        fill_value=0)

    # Group by UID and date, calculate sum of quantity and mean of price
    grouped_df = input_df.drop([price_col], axis=1).groupby([uid_col, date_col]).agg(
        {quantity_col: 'sum'}).reset_index()

    # Take the most common price
    most_common_price = input_df.groupby([uid_col, date_col])[price_col].agg(
        lambda x: x.mode().iloc[0]).reset_index()

    # Merge with most common price

    return (grouped_df.merge(most_common_price,
                                 on=[uid_col, date_col])
                                 .reset_index()
                                 .sort_values(by=[uid_col, date_col]))

def preprocess_by_price(input_df: pd.DataFrame,
                      uid_col: str = 'uid',
                      date_col: str = 'date',
                      price_col: str = 'price',
                      quantity_col: str = 'units',
                      weights_col: str = 'days',
                      ) ->tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the DataFrame by grouping data by price normalization."""
    df_by_price_norm = (input_df.groupby([uid_col, price_col])
                    .agg({quantity_col: 'mean', date_col:'count'})
                    .reset_index().sort_values(by=[uid_col, price_col]))

    df_by_price_norm[quantity_col] = df_by_price_norm[quantity_col] + 0.001

    return df_by_price_norm.rename(columns={date_col: weights_col})


def uid_with_price_changes(input_df: pd.DataFrame,
                       price_changes: int = 4, threshold: float = 0.01,
                       uid_col: str = 'uid', date_col: str = 'date',
                       price_col: str = 'price',
                       quantity_col: str = 'units') -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria."""
    # Preprocess data
    df_by_price = preprocess_by_price(input_df=input_df,
                                   uid_col=uid_col,
                                   date_col=date_col,
                                   price_col=price_col,
                                   quantity_col=quantity_col)

    # Calculate price change percentage
    df_by_price = df_by_price.sort_values(by=[uid_col, price_col])
    df_by_price['price_change'] = df_by_price.groupby(uid_col)[price_col].pct_change()

    # Filter user IDs with significant price changes
    return df_by_price[df_by_price['price_change'].abs() > threshold].groupby(
        uid_col)[price_col].nunique().loc[lambda x: x > price_changes].index.tolist()

def uid_with_min_conversions(input_df: pd.DataFrame,
                             min_days_with_conversions: int = 10,
                             uid_col: str = 'uid',
                             quantity_col: str = 'units') -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria.

    Returns:
        DataFrame: Filtered DataFrame containing elasticity candidates.
    """
    return input_df[input_df[quantity_col] > 0].groupby(
        uid_col).filter(
            lambda x: len(x) >= min_days_with_conversions)[uid_col].unique()



def get_elasticity_candidates(input_df: pd.DataFrame,
                              nb_months: int = 3, min_conversion_day_per_month: int = 10,
                              price_changes: int = 4, threshold: float = 0.01,
                              uid_col: str = 'uid', date_col: str = 'date',
                              price_col: str = 'price_from_revenue',
                              quantity_col: str = 'units') -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria.

    Args:
        input_df (DataFrame): Input DataFrame containing user interactions.
        nb_months (int): Number of months to consider for conversion.
        min_conversion_day_per_month (int): Minimum conversion days per month.
        price_changes (int): Minimum number of price changes.
        threshold (float): Minimum percentage change to consider significant.
        uid_col (str): Column name for user IDs.
        date_col (str): Column name for dates.
        price_col (str): Column name for prices.
        quantity_col (str): Column name for quantity.

    Returns:
        DataFrame: Filtered DataFrame containing elasticity candidates.
    """
    logging.info('Number of rows:', input_df.shape[0])
    logging.info('Number of unique user IDs:', input_df[uid_col].nunique())
    # Get user IDs with minimum conversion days
    min_conversion_day = nb_months * min_conversion_day_per_month
    uid_min_conversion_day = input_df[input_df[quantity_col] > 0].groupby(
        uid_col).filter(
            lambda x: len(x) >= min_conversion_day)[uid_col].unique()
    input_df = input_df[input_df[uid_col].isin(uid_min_conversion_day)]
    logging.info('Number of user IDs with more than',
                 min_conversion_day,
                 'days with conversion:', min_conversion_day,
                 len(uid_min_conversion_day))
    # Preprocess data
    df_norm = preprocess_by_price(input_df=input_df,
                                    uid_col=uid_col,
                                    date_col=date_col,
                                    price_col=price_col,
                                    quantity_col=quantity_col)
    # Calculate price change percentage
    df_norm = df_norm.sort_values(by=[uid_col, price_col])
    df_norm['price_change'] = df_norm.groupby(uid_col)[price_col].pct_change()

    # Filter user IDs with significant price changes
    uid_price_changes = df_norm[df_norm['price_change'].abs() > threshold].groupby(
        uid_col)[price_col].nunique().loc[lambda x: x > price_changes].index.tolist()
    logging.info('Number of user IDs with price changes of more than',
                 threshold*100, '%:', len(uid_price_changes))
    df_gold_candidates = input_df[input_df[uid_col].isin(uid_price_changes)]
    logging.info('Number of gold candidates:', df_gold_candidates[uid_col].nunique())
    return df_gold_candidates.sort_values(by=[uid_col, date_col])



