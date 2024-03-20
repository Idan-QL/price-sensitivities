"""Module of preprocessing."""

import numpy as np
import pandas as pd
from datetime import datetime

def read_and_preprocess(client_key: str,
                        channel: str,
                        bucket: str,
                        start_date: str,
                        end_date: str,
                        dir_: str = 'data_science/datasets',
                        price_changes: int = 4, threshold: float = 0.01,
                        min_days_with_conversions: int = 15,
                        uid_col: str = 'uid',
                        price_col: str = 'round_price',
                        quantity_col: str = 'units',
                        date_col: str = 'date') ->tuple[pd.DataFrame, pd.DataFrame]:
    """Read and preprocess the DataFrame.
    TODO: add original price_col base for round_price

    Returns:
    - df_by_day (DataFrame): DataFrame grouped by day.
    """

    df = read_monthly_data(client_key=client_key,
                           channel=channel,
                           bucket=bucket,
                           start_date=start_date,
                           end_date=end_date,
                           dir_=dir_,
                           price_changes=price_changes, threshold=threshold,
                           min_days_with_conversions=min_days_with_conversions,
                           uid_col=uid_col,
                           price_col=price_col,
                           quantity_col=quantity_col)
    
    df_by_price = preprocess_by_price(df,
                                      uid_col=uid_col,
                                      date_col=date_col,
                                      price_col=price_col,
                                      quantity_col=quantity_col)

    return df_by_price


def round_down_to_nearest_half(number: float) -> float:
    """Rounds down to the nearest half."""
    nearest_integer = int(number)
    decimal_part = number - nearest_integer
    if decimal_part < 0.5:
        nearest_half = 0
    else:
        nearest_half = 0.5
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
    else:
        if price < 10:
            return round_down_to_nearest_half(price * 10) / 10
        elif price < 50:
            return round_down_to_nearest_half(price)
        elif price > 500:
            return int(round_down_to_nearest_half(price / 10) * 10)
        else:
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
    df_by_day = (grouped_df.merge(most_common_price,
                                 on=[uid_col, date_col])
                                 .reset_index()
                                 .sort_values(by=[uid_col, date_col]))

    return df_by_day

def preprocess_by_price(input_df: pd.DataFrame,
                      uid_col: str = 'uid',
                      date_col: str = 'date',
                      price_col: str = 'price',
                      quantity_col: str = 'units',
                      weights_col: str = 'days',
                      ) ->tuple[pd.DataFrame, pd.DataFrame]:
    
    df_by_price_norm = (input_df.groupby([uid_col, price_col])
                    .agg({quantity_col: 'mean', date_col:'count'})
                    .reset_index().sort_values(by=[uid_col, price_col]))
    
    df_by_price_norm[quantity_col] = df_by_price_norm[quantity_col] + 0.001

    df_by_price_norm.rename(columns={date_col: weights_col}, inplace=True)

    return df_by_price_norm


def read_monthly_data(client_key: str, channel: str, bucket: str,
                      start_date: str, end_date: str,
                      dir_: str = 'data_science/datasets',
                      price_changes: int = 4, threshold: float = 0.01,
                      min_days_with_conversions: int = 15,
                      uid_col: str = 'uid',
                      price_col: str = 'round_price',
                      quantity_col: str = 'units'
                      ) -> pd.DataFrame:
    """Read monthly data and concatenate into a single DataFrame."""
    df_full_list = []
    uid_ok = []
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    df_ko = pd.DataFrame()  # Initialize df_ko
    while end_date_dt >= start_date_dt:
        print(f'reading {end_date_dt}')
        df_part = read_data(client_key, channel, bucket, date=end_date_dt.strftime('%Y-%m-%d'), dir_=dir_)
        df_part = df_part[~df_part['uid'].isin(uid_ok)]
        
        # Concatenate df_part with df_ko
        df_part = pd.concat([df_part, df_ko])
        
        uid_changes = uid_with_price_changes(df_part,
                                             price_changes=price_changes,
                                             threshold=threshold,
                                             price_col=price_col,
                                             quantity_col=quantity_col)
        uid_conversions = uid_with_min_conversions(df_part,
                                                   min_days_with_conversions=min_days_with_conversions,
                                                   uid_col=uid_col,
                                                   quantity_col=quantity_col)
        uid_intersection_change_conversions = list(set(uid_changes) & set(uid_conversions))
        uid_ok.extend(uid_intersection_change_conversions)

        df_ok = df_part[df_part['uid'].isin(uid_intersection_change_conversions)]
        print(f'number of uid ok: {len(uid_ok)}')
        df_ko = df_part[~df_part['uid'].isin(uid_intersection_change_conversions)]
        # print(f'number of uid not ok: {df_ko['uid'].nunique()}')

        df_full_list.append(df_ok)
        end_date_dt -= pd.DateOffset(months=1)

    result_df = pd.concat(df_full_list)
    print(f'Number of unique user IDs: {result_df["uid"].nunique()}')
    return result_df



def read_data(client_key: str, channel: str, bucket: str,
                      date: str = '2023-02-01',
                      dir_: str = 'data_science/datasets') -> pd.DataFrame:
    """Read monthly data and concatenate into a single DataFrame."""
    year_, month_ = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m').split("-")
    try:
        df = pd.read_parquet(f's3://{bucket}/{dir_}/{client_key}/{channel}/elasticity/{year_}_{int(month_)}_full_data.parquet/',
                                    columns=['date',
                                            'uid',
                                            'conversions_most_common_shelf_price',
                                            'views_most_common_shelf_price',
                                            'total_units',
                                            'price'])
        df = process_data(df)
    except:
        print(f'No data for {year_}_{int(month_)}')
        print(f's3://{bucket}/{dir_}/{client_key}/{channel}/elasticity/{year_}_{int(month_)}_full_data.parquet/')
        pass
    return df



def uid_with_price_changes(input_df: pd.DataFrame,
                       price_changes: int = 4, threshold: float = 0.01,
                       uid_col: str = 'uid', date_col: str = 'date',
                       price_col: str = 'price',
                       quantity_col: str = 'units') -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria.
    """
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
    uid_with_price_changes = df_by_price[df_by_price['price_change'].abs() > threshold].groupby(
        uid_col)[price_col].nunique().loc[lambda x: x > price_changes].index.tolist()
    return uid_with_price_changes

def uid_with_min_conversions(input_df: pd.DataFrame,
                             min_days_with_conversions: int = 10,
                             uid_col: str = 'uid', 
                             quantity_col: str = 'units') -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria.

    Returns:
        DataFrame: Filtered DataFrame containing elasticity candidates.
    """
    uid_min_days_with_conversion = input_df[input_df[quantity_col] > 0].groupby(
        uid_col).filter(
            lambda x: len(x) >= min_days_with_conversions)[uid_col].unique()
    return uid_min_days_with_conversion



def process_data(df_full: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional columns based on the read monthly data."""
    df_full['price_merged'] = np.where(~df_full.conversions_most_common_shelf_price.isna(),
                                       df_full.conversions_most_common_shelf_price,
                                       np.where(~df_full.views_most_common_shelf_price.isna(),
                                                df_full.views_most_common_shelf_price,
                                                df_full.price))

    df_full['source'] = np.where(~df_full.conversions_most_common_shelf_price.isna(),
                                 'conversions',
                                 np.where(~df_full.views_most_common_shelf_price.isna(),
                                          'views',
                                          'price_recommendations'))

    df_full['units'] = np.where(~df_full.total_units.isna(), df_full.total_units, 0)
    df_full['round_price'] = df_full['price_merged'].apply(round_price_effect)
    round_price_effect

    return df_full[['date','uid','round_price','units','price_merged','source']]

def grouped_months(client: str, channel: str, bucket: str,
                   start_date: str = '2023-12-01', end_date: str = '2024-02-29',
                   dir_: str = 'data_science/datasets') -> pd.DataFrame:
    """Group the data by month and year."""
    df_full = read_monthly_data(client, channel, bucket, start_date, end_date, dir_)
    return df_full

def get_elasticity_candidates(input_df: pd.DataFrame,
                              nb_months: int = 3, min_conversion_day_per_month: int = 10,
                              price_changes: int = 4, threshold: float = 0.01,
                              uid_col: str = 'uid', date_col: str = 'date',
                              price_col: str = 'price_from_revenue',
                              quantity_col: str = 'units') -> pd.DataFrame:
    """Filter the DataFrame to obtain elasticity candidates based on specified criteria.
    TO DO: change with smaller functions

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
    print('Number of unique user IDs:', input_df[uid_col].nunique())
    # Get user IDs with minimum conversion days
    min_conversion_day = nb_months * min_conversion_day_per_month
    uid_min_conversion_day = input_df[input_df[quantity_col] > 0].groupby(
        uid_col).filter(
            lambda x: len(x) >= min_conversion_day)[uid_col].unique()
    input_df = input_df[input_df[uid_col].isin(uid_min_conversion_day)]
    print(f'Number of user IDs with more than {min_conversion_day} days with conversion:',
          min_conversion_day, len(uid_min_conversion_day))
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
    print(f'User IDs with price changes of more than {threshold*100}%:', len(uid_price_changes))
    df_gold_candidates = input_df[input_df[uid_col].isin(uid_price_changes)]
    print("Number of gold candidates:", df_gold_candidates[uid_col].nunique())
    return df_gold_candidates.sort_values(by=[uid_col, date_col])



