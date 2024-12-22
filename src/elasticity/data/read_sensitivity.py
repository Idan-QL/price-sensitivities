"""This module reads data from data-lake for a given client and channel."""

import logging
from typing import Optional

import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from elasticity.data.utils import calculate_date_range, extract_date_params
from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.data_catalog.athena_query_manager import AthenaQuery


def read_data_query(
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
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
    logging.info(
        f"Reading data for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"source: {data_fetch_params.source}; "
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    file_name = f"{app_state.project_name}_sensitivity"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
    )

    data_df = athena_query.execute_query()
    data_df["uid_competitor_name"] = (
        data_df["uid"].astype(str).str.cat(data_df["competitor_name"].astype(str), sep="_")
    )
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df


def read_top_competitors(
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
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
    logging.info(
        f"Reading data for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"source: {data_fetch_params.source}; "
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    file_name = f"{app_state.project_name}_sensitivity_competitors"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
    )

    data_df = athena_query.execute_query()
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df


def read_data_query_per_uid(
    uid: str,
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Read data from data-lake for a given client and channel.

    Args:
        uid (str) : uid string.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_params (Optional[dict], optional): Date parameters for filtering the data.
        Defaults to None.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    days_back, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(days_back, start_date, end_date)
    logging.info(
        f"Reading data for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"source: {data_fetch_params.source}; "
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    file_name = f"{app_state.project_name}_sensitivity_per_uid"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        uid=uid,
        start_date=start_date,
        end_date=end_date,
    )

    data_df = athena_query.execute_query()
    data_df["uid_competitor_name"] = (
        data_df["uid"].astype(str).str.cat(data_df["competitor_name"].astype(str), sep="_")
    )
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df


def read_data_query_per_competitor(
    competitor_name: str,
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Read data from data-lake for a given client and channel.

    Args:
        competitor_name (str): Competitor name.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_params (Optional[dict], optional): Date parameters for filtering the data.
        Defaults to None.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    days_back, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(days_back, start_date, end_date)
    logging.info(
        f"Reading data for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"source: {data_fetch_params.source}; "
        f"start_date: {start_date};"
        f"end_date: {end_date} ..."
    )

    file_name = f"{app_state.project_name}_sensitivity_per_competitor"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        competitor_name=competitor_name,
    )

    data_df = athena_query.execute_query()
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df
