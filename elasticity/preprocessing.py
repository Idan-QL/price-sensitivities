"""Module of preprocessing."""

import pandas as pd


def preprocess_by_day(input_df: pd.DataFrame,
                      uid_col: str = 'uid',
                      date_col: str = 'date',
                      price_col: str = 'price_from_revenue',
                      quantity_col: str = 'units',
                      median_price_col: str = 'uid_price_from_revenue_median',
                      median_quantity_col: str = 'median_sale_per_day'
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

    # Calculate the median sale per day by UID and merge it with the original DataFrame
    average_median_per_day = pivot_df.median(axis=1).rename('median_sale_per_day')
    input_df = input_df.merge(average_median_per_day, left_on=uid_col, right_index=True)

    input_df['uid_price_from_revenue_median'] = input_df.groupby(uid_col)[price_col].transform(
        'median')
    input_df['price_delta_from_median'] = (input_df[price_col]
                                           - input_df['uid_price_from_revenue_median'])
    input_df['day_sales_delta_from_median'] = (input_df.groupby([uid_col, date_col])
                                               [quantity_col].transform('sum')
                                               - input_df['median_sale_per_day'])

    # Group by UID and date, calculate sum of quantity and mean of price
    grouped_df = input_df.drop([price_col], axis=1).groupby([uid_col, date_col]).agg(
        {quantity_col: 'sum', median_price_col: 'first', median_quantity_col:'first'}).reset_index()

    # Take the most common price
    most_common_price = input_df.groupby([uid_col, date_col])[price_col].agg(
        lambda x: x.mode().iloc[0]).reset_index()

    # Merge with most common price
    df_by_day = (grouped_df.merge(most_common_price,
                                 on=[uid_col, date_col])
                                 .reset_index()
                                 .sort_values(by=[uid_col, date_col]))

    df_by_price_norm = (df_by_day.groupby([uid_col, price_col])
                        .agg({quantity_col: 'mean',
                              median_price_col: 'first',
                              median_quantity_col:'first',
                              date_col:'count'})
                        .reset_index().sort_values(by=[uid_col, price_col]))

    return df_by_day, df_by_price_norm
