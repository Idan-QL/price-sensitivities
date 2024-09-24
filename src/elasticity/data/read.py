"""This module reads data from data-lake for a given client and channel."""

import logging
from typing import Optional

import pandas as pd

from elasticity.data.configurator import DataColumns, DataFetchParameters
from elasticity.data.utils import calculate_date_range, extract_date_params
from ql_toolkit.data_lake.athena_query import AthenaQuery


def read_data_query(
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
    filter_units: Optional[bool] = False,
    data_columns: Optional[DataColumns] = None,
) -> pd.DataFrame:
    """Read data from data-lake for a given client and channel.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_params (Optional[dict], optional): Date parameters for filtering the data.
        Defaults to None.
        filter_units (Optional[bool], optional): Whether to filter units greater than 0.
        Defaults to False.
        data_columns (Optional[DataColumns]): Configuration of data columns.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    days_back, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(days_back, start_date, end_date)
    data_columns = data_columns or DataColumns()
    logging.info(
        f"Reading data from data-lake for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    filter_units_condition_datalake = (
        "AND (CASE WHEN product_events_analytics['units_sold'] IS NOT NULL "
        "THEN CAST(product_events_analytics['units_sold'] AS DOUBLE) "
        "ELSE CAST(product_info_analytics['units_sold'] AS DOUBLE) END > 0 )"
    )

    filter_units_condition_elasticity = "AND units > 0"

    filter_units_condition = (
        (
            filter_units_condition_datalake
            if data_fetch_params.read_from_datalake
            else filter_units_condition_elasticity
        )
        if filter_units
        else ""
    )

    file_name = f"elasticity{'_datalake' if data_fetch_params.read_from_datalake else ''}"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        filter_units_condition=filter_units_condition,
        data_columns=data_columns,
    )

    data_df = athena_query.execute_query()
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df


def read_data_attrs(
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Read 3 months attrs data from data-lake for a given client and channel.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_params (dict): The dictionary containing date parameters:(optional)
        - days_back (int): The number of days back to read data. Default is 7 days.
        - end_date (str): The end date for the query in 'YYYY-MM-DD' format. Default is today.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    _, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(90, start_date, end_date)
    logging.info(
        f"Reading attrs data from data-lake for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )
    attr_names_str = ", ".join(f"'{attr}'" for attr in [data_fetch_params.attr_name])

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name="elasticity_attrs",
        start_date=start_date,
        end_date=end_date,
        attr_names_str=attr_names_str,
    )

    df_attrs = athena_query.execute_query()
    logging.info(f"Finishing reading attrs data from data-lake. Shape: {df_attrs.shape}")
    return df_attrs
