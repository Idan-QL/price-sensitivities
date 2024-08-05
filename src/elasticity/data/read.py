"""This module reads data from data-lake for a given client and channel."""

import logging
from typing import Optional

import pandas as pd

from elasticity.data.utils import calculate_date_range, extract_date_params
from ql_toolkit.data_lake.athena_query import AthenaQuery


def read_data_query(
    client_key: str,
    channel: str,
    date_params: Optional[dict] = None,
    filter_units: Optional[bool] = False,
) -> pd.DataFrame:
    """Read data from data-lake for a given client and channel.

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        date_params (Optional[dict], optional): Date parameters for filtering the data.
        Defaults to None.
        filter_units (Optional[bool], optional): Whether to filter units greater than 0.
        Defaults to False.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    days_back, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(days_back, start_date, end_date)
    logging.info(
        f"Reading data from data-lake for client: {client_key}; "
        f"channel: {channel};"
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    filter_units_condition = "AND units > 0" if filter_units else ""

    athena_query = AthenaQuery(
        client_key=client_key,
        channel=channel,
        file_name="elasticity",
        start_date=start_date,
        end_date=end_date,
        filter_units_condition=filter_units_condition,
    )

    data_df = athena_query.execute_query()
    logging.info(f"Finishing reading attrs data from data-lake. Shape: {data_df.shape}")
    return data_df


def read_data_attrs(
    client_key: str,
    channel: str,
    attr_names: list,
    date_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Read 3 months attrs data from data-lake for a given client and channel.

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        date_params (dict): The dictionary containing date parameters:(optional)
        - days_back (int): The number of days back to read data. Default is 7 days.
        - end_date (str): The end date for the query in 'YYYY-MM-DD' format. Default is today.
        attr_names (list): The list of attribute names.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    _, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(90, start_date, end_date)
    logging.info(
        f"Reading attrs data from data-lake for client: {client_key}; "
        f"channel: {channel};"
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )
    attr_names_str = ", ".join(f"'{attr}'" for attr in attr_names)

    athena_query = AthenaQuery(
        client_key=client_key,
        channel=channel,
        file_name="elasticity_attrs",
        start_date=start_date,
        end_date=end_date,
        attr_names_str=attr_names_str,
    )

    df_attrs = athena_query.execute_query()
    logging.info(f"Finishing reading attrs data from data-lake. Shape: {df_attrs.shape}")
    return df_attrs
