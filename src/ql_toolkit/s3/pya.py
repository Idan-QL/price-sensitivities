"""This module contains functions to query from pyathena."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import pandas as pd
from pyathena import connect
from pyathena.error import DatabaseError, OperationalError
from pyathena.pandas.cursor import PandasCursor

from ql_toolkit.config.runtime_config import app_state


def calculate_date_range(
    days_back: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[str, str]:
    """Calculate the start and end date based on the given days_back, start_date, and end_date.

    Args:
        days_back (int): The number of days back to calculate the start date.
        Ignored if start_date is provided.
        start_date (str): The start date for the calculation in 'YYYY-MM-DD' format.
        end_date (str): The end date for the calculation in 'YYYY-MM-DD' format.
        Default is yesterday.

    Returns:
        tuple: A tuple containing the start date and end date in 'YYYY-MM-DD' format.

    Example:
        >>> calculate_date_range(days_back=7, end_date='2023-06-19')
        ('2023-06-12', '2023-06-19')
        >>> calculate_date_range(start_date='2023-06-12', end_date='2023-06-19')
        ('2023-06-12', '2023-06-19')
    """
    if start_date is not None and days_back is not None:
        error_message = (
            "Both start_date and days_back cannot be provided simultaneously."
        )
        raise ValueError(error_message)

    end_date = (
        datetime.now(timezone.utc) - timedelta(days=1)
        if end_date is None
        else datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    )

    if start_date is None:
        if days_back is None:
            error_message = "Either days_back or start_date must be provided."
            raise ValueError(error_message)
        start_date = end_date - timedelta(days=days_back)
    else:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )

    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")

    return start_date, end_date


def extract_date_params(
    date_params: Optional[Dict[str, Optional[str]]] = None
) -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """Extract days_back, start_date, and end_date from the date_params dictionary.

    Args:
        date_params (dict): The dictionary containing date parameters.Default None

    Returns:
        tuple: A tuple containing days_back, start_date, and end_date.
    """
    if date_params is None:
        date_params = {}
    days_back = date_params.get("days_back", None)
    start_date = date_params.get("start_date", None)
    end_date = date_params.get("end_date", None)

    # Adjust days_back to None if start_date is provided
    if start_date is not None and days_back is not None:
        error_message = (
            "Both start_date and days_back cannot be provided simultaneously."
        )
        raise ValueError(error_message)
    if start_date is not None and days_back is None:
        days_back = None
    elif start_date is None and days_back is None:
        days_back = 7  # Default days_back if neither are provided

    return days_back, start_date, end_date


def execute_query(query: str, params: dict) -> pd.DataFrame:
    """Execute the given query with the specified parameters and return a DataFrame.

    Args:
        query (str): The SQL query to execute.
        params (dict): The parameters for the SQL query.

    Returns:
        DataFrame: A DataFrame containing the results of the query.
    """
    try:
        cursor = connect(
            s3_staging_dir=app_state.s3_athena_dir,
            region_name=app_state.s3_region,
            cursor_class=PandasCursor,
        ).cursor()
        return cursor.execute(query, parameters=params).as_pandas()
    except (DatabaseError, OperationalError) as e:
        logging.error(f"Database error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return pd.DataFrame()


def pivot_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the attributes DataFrame to have one row per uid with attributes as columns.

    Args:
        df (DataFrame): The DataFrame containing attribute data.

    Returns:
        DataFrame: A pivoted DataFrame with attributes as columns.

    Example:
        >>> df = pd.DataFrame({'uid': [123, 123],
                               'attr_name': ['attr1', 'attr2'],
                               'attr_value': ['value1', 'value2']})
        >>> pivot_attributes(df)
           uid attr1 attr2
        0  123 value1 value2
    """
    try:
        return df.pivot_table(
            index="uid", columns="attr_name", values="attr_value", aggfunc="first"
        ).reset_index()
    except Exception as e:
        logging.error(f"Error pivoting DataFrame: {e}")
        return pd.DataFrame()


def query_client_attrs(
    client_key: str,
    channel: str,
    attr_names: list,
    date_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Query client attrs for a specific client key and attribute names.

    within a specified number of days.

    Args:
        client_key (str): The client key to process data for.
        channel (str): The channel to filter data for.
        attr_names (list): The list of attribute names.
        date_params (dict): The dictionary containing date parameters:(optional)
        - days_back (int): The number of days back to read data. Default is 7 days.
        - end_date (str): The end date for the query in 'YYYY-MM-DD' format. Default is today.

    Returns:
        DataFrame: A DataFrame containing the most recent attributes for each uid.

    Example:
        >>> client_key = "example_client"
        >>> channel = "example_channel"
        >>> attr_names = ["attr1", "attr2"]
        >>> query_client_attrs(client_key, channel, attr_names)
            uid        date  attr_name attr_value
        0  123  2023-06-12     attr1     value1
        1  123  2023-06-12     attr2     value2
    """
    days_back, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(days_back, start_date, end_date)
    table_name = f"AwsDataCatalog.analytics.client_key_{client_key}"
    attr_names_str = ", ".join(f"'{attr}'" for attr in attr_names)

    query = f"""
    WITH ranked_attrs AS (
        SELECT
            uid,
            date,
            t.element.name AS attr_name,
            t.element.value AS attr_value,
            ROW_NUMBER() OVER (PARTITION BY uid, t.element.name ORDER BY date DESC) AS rn
        FROM {table_name}, UNNEST(attrs) AS t(element)
        WHERE t.element.name IN ({attr_names_str})
            AND (t.element.value IS NOT NULL OR CAST(t.element.value AS VARCHAR) != '')
            AND date BETWEEN %(start_date)s AND %(end_date)s
            AND channel = %(channel)s
    )
    SELECT uid, date, attr_name, attr_value
    FROM ranked_attrs
    WHERE rn = 1;
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "channel": channel,
    }

    df_attrs = execute_query(query, params)

    return pivot_attributes(df_attrs)


# TODO: for cov add MAX(competitors) AS competitors
def query_agg_day_client_data(
    client_key: str,
    channel: str,
    date_params: Optional[dict] = None,
    filter_units: Optional[bool] = False,
) -> pd.DataFrame:
    """Query aggregated daily client data from the AWS data catalog.

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

    table_name = f"AwsDataCatalog.analytics.client_key_{client_key}"

    where_clause = """
    WHERE date BETWEEN %(start_date)s AND %(end_date)s
        AND channel = %(channel)s
    """
    if filter_units:
        where_clause += " AND units > 0"

    query = f"""
    SELECT uid,
           date,
           AVG(shelf_price) AS shelf_price,
           SUM(units) AS units,
           SUM(revenue) AS revenue,
           MAX(inventory) AS inventory
    FROM {table_name}
    {where_clause}
    GROUP BY
    uid, date;
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "channel": channel,
    }

    return execute_query(query, params)


def query_client_data(
    client_key: str,
    channel: str,
    data_names: list,
    date_params: Optional[dict] = None,
    filter_units: Optional[bool] = False,
) -> pd.DataFrame:
    """Query client data for a specific client key and data names within a specified number of days.

    Args:
        client_key (str): The client key to process data for.
        channel (str): The channel to filter data for.
        data_names (list): The list of data names.
        date_params (dict): The dictionary containing date parameters:(optional)
        - days_back (int): The number of days back to read data. Default is 7 days.
        - end_date (str): The end date for the query in 'YYYY-MM-DD' format. Default is today.
        filter_units (bool): Whether to filter results where units > 0. Default is False.

    Returns:
        DataFrame: A DataFrame containing the processed data for the client key and data names.

    Example:
        >>> client_key = "example_client"
        >>> channel = "example_channel"
        >>> attr_names = ["attr1", "attr2"]
        >>> date_params = {"days_back": 7, "end_date": "2023-06-19"}
        >>> query_client_attrs(client_key, channel, attr_names, date_params)
            uid        date  attr_name attr_value
        0  123  2023-06-12     attr1     value1
        1  123  2023-06-12     attr2     value2
    """
    days_back, start_date, end_date = extract_date_params(date_params)
    start_date, end_date = calculate_date_range(days_back, start_date, end_date)

    table_name = f"AwsDataCatalog.analytics.client_key_{client_key}"

    data_names_str = ", ".join(f"{data}" for data in data_names)

    where_clause = """
    WHERE date BETWEEN %(start_date)s AND %(end_date)s
        AND channel = %(channel)s
    """
    if filter_units:
        where_clause += " AND units > 0"

    query = f"""
    SELECT uid,
           date,
           {data_names_str}
    FROM {table_name}
    {where_clause};
    """

    params = {
        "start_date": start_date,
        "end_date": end_date,
        "channel": channel,
    }

    return execute_query(query, params)
