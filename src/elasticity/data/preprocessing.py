"""Module of preprocessing."""

import logging
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from elasticity.data.configurator import DataColumns, PreprocessingParameters
from elasticity.data.utils import (
    get_revenue,
    initialize_dates,
    outliers_iqr_filtered,
    preprocess_by_price,
    round_price_effect,
    uid_with_min_conversions,
    uid_with_price_changes,
)
from ql_toolkit.config.runtime_config import app_state


def read_and_preprocess(
    client_key: str,
    channel: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    uids_to_filter: Optional[list] = None,
    preprocessing_parameters: Optional[PreprocessingParameters] = None,
    data_columns: Optional[DataColumns] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[int],
    Optional[str],
    Optional[pd.DataFrame],
    Optional[int],
]:
    """Reads and preprocesses the DataFrame.

    Args:
        client_key (str): Client key for data retrieval.
        channel (str): Channel identifier.
        start_date (str, optional): Start date for data retrieval. Defaults to None.
        end_date (str, optional): End date for data retrieval. Defaults to None.
        uids_to_filter (list, optional): UID filter for data retrieval. Defaults to None.
        preprocessing_parameters (PreprocessingParameters, optional): Preprocessing parameters.
        Defaults to PreprocessingParameters().
        data_columns (DataColumns, optional): Data columns configuration. Defaults to DataColumns().

    Returns:
        tuple: A tuple containing:
            - df_by_price (pd.DataFrame): DataFrame grouped by price.
            - df (pd.DataFrame): Original DataFrame.
            - total_uid (int): Total number of unique IDs.
            - end_date (str): End date used for data retrieval.
            - df_revenue_uid (pd.DataFrame): DataFrame containing revenue and UID.
            - total_revenue (int): Total revenue.
            Returns None for these values in case of error.
    """
    # Initialize default parameters if not provided
    if preprocessing_parameters is None:
        preprocessing_parameters = PreprocessingParameters()
    if data_columns is None:
        data_columns = DataColumns()

    try:
        # Initialize dates and validate formats
        start_date, end_date = initialize_dates(start_date, end_date)
        logging.info(f"start_date: {start_date}")
        logging.info(f"end_date: {end_date}")

        # Data retrieval with error handling
        try:
            df, total_uid, df_revenue_uid, total_revenue = (
                progressive_monthly_aggregate(
                    client_key=client_key,
                    channel=channel,
                    start_date=start_date,
                    end_date=end_date,
                    uids_to_filter=uids_to_filter,
                    preprocessing_parameters=preprocessing_parameters,
                    data_columns=data_columns,
                )
            )
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None, None, None, end_date, None, None

        # Data preprocessing
        try:
            df_by_price = preprocess_by_price(df, data_columns=data_columns)
        except Exception as e:
            logging.error(f"Error during preprocessing by price: {e}")
            return None, df, total_uid, end_date, df_revenue_uid, total_revenue

        return df_by_price, df, total_uid, end_date, df_revenue_uid, total_revenue

    except ValueError as e:
        logging.error(f"Date format error: {e}")
        return None, None, None, None, None, None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None, None, None, None, None, None


def progressive_monthly_aggregate(
    client_key: str,
    channel: str,
    start_date: str,
    end_date: str,
    uids_to_filter: Optional[str] = None,
    preprocessing_parameters: Optional[PreprocessingParameters] = None,
    data_columns: Optional[DataColumns] = None,
) -> pd.DataFrame:
    """Progressive aggregation of data.

    Perform progressive aggregation on monthly data to identify UIDs
    with sufficient conversions and price changes for elasticity calculation.

    Starting from end_date:
        - Read data
        - tag outliers
        - Save data of uids passing the min price changes and number of days with conversions
        to approved_data and uids to approved_uids
        - Save the other uids to rejected_data
        - reading next month of data, concat with rejected_data and test
        Loop until start_date to find more uid for elasticity by adding history

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        start_date (str): The start date in the format 'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.
        uids_to_filter (Optional[str], optional): UIDs to filter. Defaults to None.
        preprocessing_parameters (PreprocessingParameters, optional): Preprocessing parameters.
            Defaults to PreprocessingParameters().
        data_columns (DataColumns, optional): Data columns configuration.
            Defaults to DataColumns().

    Returns:
        Tuple[pd.DataFrame, int, pd.DataFrame, int]: A tuple containing:
            - The aggregated result dataframe.
            - Total number of unique user IDs considered.
            - DataFrame containing revenue information.
            - Total revenue.
    """
    # Initialize default parameters if not provided
    if preprocessing_parameters is None:
        preprocessing_parameters = PreprocessingParameters()
    if data_columns is None:
        data_columns = DataColumns()

    approved_data_list = []
    approved_uids = []

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    dates_list = pd.date_range(
        start=start_date_dt, end=end_date_dt, freq="MS"
    ).strftime("%Y-%m-%d")[::-1]

    uid_col = data_columns.uid
    qtity_col = data_columns.quantity
    rejected_data = pd.DataFrame()
    total_uid = 0

    for date_month in dates_list:
        logging.info(f"reading: {date_month}")

        df_month = read_data(
            client_key,
            channel,
            uids_to_filter=uids_to_filter,
            date=date_month,
        )

        df_month = process_data(df_month)

        if total_uid == 0:
            total_uid, total_revenue, df_revenue_uid = get_revenue(
                df_month, uid_col, total_uid
            )

        # filter out uid already approved
        df_month = df_month[~df_month[uid_col].isin(approved_uids)]
        # Concatenate df_month with rejected_data from previous months
        data_to_test = pd.concat([df_month, rejected_data])

        data_to_test["outlier_quantity"] = data_to_test.groupby(uid_col)[
            qtity_col
        ].transform(outliers_iqr_filtered)

        uid_changes = uid_with_price_changes(
            data_to_test,
            price_changes=preprocessing_parameters.price_changes,
            threshold=preprocessing_parameters.threshold,
            data_columns=data_columns,
        )

        uid_conversions = uid_with_min_conversions(
            data_to_test,
            min_conversions_days=preprocessing_parameters.min_conversions_days,
            uid_col=uid_col,
            quantity_col=qtity_col,
        )
        uid_intersection_change_conversions = list(
            set(uid_changes) & set(uid_conversions)
        )
        approved_uids.extend(uid_intersection_change_conversions)

        approved_data = data_to_test[
            data_to_test[uid_col].isin(uid_intersection_change_conversions)
        ]
        logging.info(f"number of uid ok: {len(approved_uids)}")
        rejected_data = data_to_test[
            ~data_to_test[uid_col].isin(uid_intersection_change_conversions)
        ]

        approved_data_list.append(approved_data)

    result_df = pd.concat(approved_data_list)
    logging.info(f"Number of unique user IDs: {result_df.uid.nunique()}")
    return result_df, total_uid, df_revenue_uid, total_revenue


def read_data(
    client_key: str,
    channel: str,
    uids_to_filter: Optional[list] = None,
    date: str = "2024-02-01",
) -> pd.DataFrame:
    """Read one month data. Filter out inventory 0. Option to filter one uids.

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        uids_to_filter (Optional[str], optional): A list of uids to filter the data.
        Defaults to None.
        date (str, optional): The date in the format "YYYY-MM-DD".
        Defaults to "2024-02-01".

    Returns:
        pd.DataFrame: The DataFrame containing the read data.

    Raises:
        Exception: If there is an error reading the data.

    """
    year_, month_ = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m").split("-")
    cs = [
        "date",
        "uid",
        "conversions_most_common_shelf_price",
        "views_most_common_shelf_price",
        "total_units",
        "price",
        "inventory",
    ]
    dataset_dir = app_state.s3_datasets_dir(client_key, channel)
    try:
        filters = (
            [("uid", "in", uids_to_filter)] if uids_to_filter is not None else None
        )
        df_read = pd.read_parquet(
            f"s3://{app_state.bucket_name}/{dataset_dir}/{year_}_{int(month_)}_full_data.parquet/",
            columns=cs,
            filters=filters,
        )
        logging.info(
            f'Number of inventory less or equal to 0: {len(df_read[df_read["inventory"] <= 0])}'
        )
        df_read = df_read[df_read["inventory"] > 0].drop(columns=["inventory"])
    except Exception:
        logging.error(f"No data for {year_!s}_{int(month_)!s}")
        df_read = pd.DataFrame(columns=cs)
        pass
    return df_read


def process_data(
    df_full: pd.DataFrame,
    data_columns: Optional[DataColumns] = None,
) -> pd.DataFrame:
    """Merge and round price.

    - If conversion price is available, use conversion price.
    - else if view price is available, use views price.
    - Otherwise, use price recommendations.
    - Round the price.

    Args:
        df_full (pd.DataFrame): The input DataFrame containing the monthly data.
        data_columns (DataColumns, optional): An instance of the DataColumns class.
        Defaults to DataColumns().

    Returns:
        pd.DataFrame: The processed DataFrame with additional columns.
    """
    if data_columns is None:
        data_columns = DataColumns()

    df_full["price_merged"] = np.where(
        ~df_full.conversions_most_common_shelf_price.isna(),
        df_full.conversions_most_common_shelf_price,
        np.where(
            ~df_full.views_most_common_shelf_price.isna(),
            df_full.views_most_common_shelf_price,
            df_full.price,
        ),
    )

    df_full["source"] = np.where(
        ~df_full.conversions_most_common_shelf_price.isna(),
        "conversions",
        np.where(
            ~df_full.views_most_common_shelf_price.isna(),
            "views",
            "price_recommendations",
        ),
    )

    df_full[data_columns.quantity] = np.where(
        ~df_full.total_units.isna(), df_full.total_units, 0
    )
    df_full[data_columns.price] = df_full["price_merged"].apply(round_price_effect)

    return df_full[
        [
            data_columns.date,
            data_columns.uid,
            data_columns.price,
            data_columns.quantity,
            "price_merged",
            "source",
        ]
    ]
