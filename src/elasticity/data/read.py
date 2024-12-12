"""This module reads data from data-lake for a given client and channel."""

import logging
from typing import Optional

import pandas as pd

from elasticity.data.configurator import DataColumns, DataFetchParameters
from elasticity.data.utils import calculate_date_range, extract_date_params
from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.data_catalog.athena_query_manager import AthenaQuery


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
        f"Reading data for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"source: {data_fetch_params.source}; "
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    filter_units_condition_datalake = (
        "AND (CASE WHEN product_events_analytics['units_sold'] IS NOT NULL "
        "THEN CAST(product_events_analytics['units_sold'] AS DOUBLE) "
        "ELSE CAST(product_info_analytics['units_sold'] AS DOUBLE) END > 0 )"
    )

    filter_units_condition_analytics = "AND units > 0"

    filter_units_condition_transaction = "AND units_sold > 0"

    filter_units_condition = (
        (
            filter_units_condition_datalake
            if data_fetch_params.source == "product_extended_daily"
            else (
                filter_units_condition_transaction
                if data_fetch_params.source == "product_transaction"
                else (
                    filter_units_condition_analytics
                    if data_fetch_params.source == "analytics"
                    else ""
                )
            )
        )
        if filter_units
        else ""
    )

    file_name = f"{app_state.project_name}_{data_fetch_params.source}"

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
    """Read 3 months of attributes data for a given client and channel.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_params (dict, optional): The dictionary containing date parameters:
            - days_back (int): The number of days back to read data. Default is 7 days.
            - end_date (str): The end date for the query in 'YYYY-MM-DD' format. Default is today.

    Returns:
        pd.DataFrame: A DataFrame containing the queried data.

    Raises:
        ValueError: If attr_names is empty or None, or if the fetched DataFrame is empty.
    """
    # Validate attr_names
    if not data_fetch_params.attr_names:
        raise ValueError("Attribute names (`attr_names`) must be provided and cannot be empty.")

    # Extract and calculate date range
    _, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(90, start_date, end_date)

    logging.info(
        f"Reading attrs data from data-lake for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel}; "
        f"start_date: {start_date}; "
        f"end_date: {end_date} ..."
    )

    # Generate SQL query components using helper functions
    attr_columns = generate_attribute_columns(data_fetch_params.attr_names)
    attr_names_str_for_analytics_only = generate_attribute_names_string(
        data_fetch_params.attr_names
    )
    attr_selects_for_analytics_only = generate_attribute_selects(data_fetch_params.attr_names)

    file_name = f"{app_state.project_name}_attrs_{data_fetch_params.source}"

    # Initialize AthenaQuery with generated SQL components
    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        attr_columns=attr_columns,
        attr_names_str_for_analytics_only=attr_names_str_for_analytics_only,
        attr_selects_for_analytics_only=attr_selects_for_analytics_only,
    )

    # Execute the Athena query to fetch data
    df_attrs = athena_query.execute_query()

    # Validate fetched DataFrame
    if df_attrs.empty:
        logging.error(
            f"No data found for client: {data_fetch_params.client_key}, "
            f"channel: {data_fetch_params.channel}, "
            f"start_date: {start_date}, end_date: {end_date}."
        )
        raise ValueError(
            f"Attrs data is empty for client {data_fetch_params.client_key} "
            f"and channel {data_fetch_params.channel} between {start_date} and {end_date}."
        )

    logging.info(f"Finishing reading attrs data from data-lake. Shape: {df_attrs.shape}")
    return df_attrs


def generate_attribute_columns(attr_names: list) -> str:
    """Generates a comma-separated string of attribute columns for the SQL query.

    Args:
        attr_names (list): List of attribute names.

    Returns:
        str: Formatted string for SQL attribute columns.
    """
    return ", ".join([f"attributes_map['{attr}'][1] AS {attr}" for attr in attr_names])


def generate_attribute_names_string(attr_names: list) -> str:
    """Generates a comma-separated string of attribute names for analytics.

    Args:
        attr_names (list): List of attribute names.

    Returns:
        str: Formatted string of attribute names for analytics.
    """
    return ", ".join(f"'{attr}'" for attr in attr_names)


def generate_attribute_selects(attr_names: list) -> str:
    """Generates SQL SELECT statements for each attribute for analytics.

    Args:
        attr_names (list): List of attribute names.

    Returns:
        str: Formatted string of SQL SELECT statements for analytics.
    """
    return ",\n    ".join(
        [
            f"MAX(CASE WHEN attr_name = '{attr}' THEN attr_value END) AS {attr}"
            for attr in attr_names
        ]
    )
