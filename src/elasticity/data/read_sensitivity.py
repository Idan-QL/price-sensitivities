"""This module reads data from data-lake for a given client and channel."""

import calendar
import logging
from typing import Optional

import pandas as pd

from elasticity.data.configurator import DataColumns, DataFetchParameters, DateRange
from elasticity.data.utils import (
    calculate_date_range,
    extract_date_params,
    parse_data_month,
)
from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.data_catalog.athena_query_manager import AthenaQuery


def read_top_competitors(
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
) -> pd.DataFrame:
    """Read data from data-lake for a given client and channel.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_range (DateRange): Date parameters for filtering the data.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    last_data_month = parse_data_month(date_range.end_date)
    date_range_end_month = DateRange(
        start_date=str(last_data_month.replace(day=1)),
        end_date=str(
            last_data_month.replace(
                day=calendar.monthrange(last_data_month.year, last_data_month.month)[1]
            )
        ),
    )
    logging.info(
        f"Reading data for client: {data_fetch_params.client_key}; "
        f"channel: {data_fetch_params.channel};"
        f"source: {data_fetch_params.source}; "
        f"start_date: {date_range_end_month.start_date};"
        f"end_date: {date_range_end_month.end_date} ..."
    )

    file_name = f"{app_state.project_name}_competitors"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=date_range_end_month.start_date,
        end_date=date_range_end_month.end_date,
    )

    data_df = athena_query.execute_query()
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df


def read_data_query_per_competitor(
    data_fetch_params: DataFetchParameters,
    date_params: Optional[dict] = None,
    data_columns: DataColumns = None,
) -> pd.DataFrame:
    """Read data from data-lake for a given client and channel.

    Args:
    data_fetch_params : DataFetchParameters
        Parameters related to data fetching, including client key, channel, and competitor name.
    date_params : Optional[dict], optional
        Date parameters for filtering the data. Defaults to None.
    data_columns : Optional[DataColumns], optional
        Columns to include in the fetched data. Defaults to None.

    Returns:
        DataFrame: A DataFrame containing the queried data.
    """
    if not data_fetch_params.competitor_name:
        logging.error("No competitor_name. Return empty df")
        return pd.DataFrame()

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

    file_name = f"{app_state.project_name}_per_competitor"

    athena_query = AthenaQuery(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        file_name=file_name,
        start_date=start_date,
        end_date=end_date,
        competitor_name=data_fetch_params.competitor_name,
        data_columns=data_columns,
    )

    data_df = athena_query.execute_query()
    logging.info(f"Finishing reading data from data-lake. Shape: {data_df.shape}")
    return data_df
