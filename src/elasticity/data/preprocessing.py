"""Module of preprocessing."""

import calendar
import logging
import warnings
from typing import List, Optional, Tuple

import pandas as pd
from pydantic import ValidationError

from elasticity.data.configurator import (
    DataColumns,
    DataFetchParameters,
    DateRange,
    PreprocessingParameters,
)
from elasticity.data.read import read_data_query
from elasticity.data.utils import (
    calculate_last_values,
    clean_inventory_data,
    clean_quantity_data,
    create_date_list,
    filter_data_by_uids,
    get_revenue,
    get_uid_changes_and_conversions,
    initialize_dates,
    log_rejection_reasons,
    outliers_iqr_filtered,
    parse_data_month,
    preprocess_by_price,
    round_price_effect,
)
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3 import io_tools as s3io

# Suppress only the specific divide by zero warning
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide"
)


def run_preprocessing(
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
    data_columns: DataColumns,
    csv_path: Optional[str] = None,
) -> tuple:
    """Preprocess and load the required data.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching such as
        client key, channel, and attribute name.
        date_range (DateRange): The date range for fetching the data.
        data_columns (DataColumns): Configuration for column mappings used in the
        preprocessing step.
        csv_path (Optional[str]): Path to a CSV file. If provided, reads data from CSV instead.

    Returns:
        tuple: A tuple containing:
            - df_by_price (pd.DataFrame): The DataFrame containing price data.
            - df_revenue_uid (pd.DataFrame): The DataFrame containing revenue data by UID.
            - total_end_date_uid (int): The total number of UIDs for the end date.
            - total_revenue (float): The total revenue calculated from the data.
    """
    df_by_price, _, total_end_date_uid, df_revenue_uid, total_revenue = read_and_preprocess(
        data_fetch_params=data_fetch_params,
        date_range=date_range,
        data_columns=data_columns,
        csv_path=csv_path,
    )

    if df_by_price is None or df_by_price.empty:
        raise ValueError("Error: df_by_price is None or Empty")

    return df_by_price, df_revenue_uid, total_end_date_uid, total_revenue


def save_preprocess_to_s3(
    df_by_price: pd.DataFrame,
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
    is_qa_run: bool,
) -> None:
    """Save the processed data to S3.

    Args:
        df_by_price (pd.DataFrame): The DataFrame containing price data.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching
        such as client key, channel, and attribute name.
        date_range (DateRange): The date range for fetching the data.
        is_qa_run (bool): Flag indicating if the script is running in QA environment.

    Returns:
        None
    """
    file_name_suffix = "_qa" if is_qa_run else ""
    file_name = (
        f"df_by_price_{data_fetch_params.client_key}_{data_fetch_params.channel}_"
        f"{date_range.end_date}{file_name_suffix}.parquet"
    )

    s3io.write_dataframe_to_s3(
        file_name=file_name,
        xdf=df_by_price,
        s3_dir=app_state.s3_eval_results_dir,
    )


def read_and_preprocess(
    data_fetch_params: DataFetchParameters,
    date_range: Optional[DateRange] = None,
    preprocessing_parameters: Optional[PreprocessingParameters] = None,
    data_columns: Optional[DataColumns] = None,
    csv_path: Optional[str] = None,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[int],
    Optional[pd.DataFrame],
    Optional[int],
]:
    """Reads and preprocesses data, handling errors and logging information throughout the process.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_range (Optional[DateRange]): Date range for filtering data.
        preprocessing_parameters (Optional[PreprocessingParameters]): Parameters for preprocessing.
        data_columns (Optional[DataColumns]): Configuration of data columns.
        csv_path (Optional[str]): Path to a CSV file. If provided, reads data from CSV instead.

    Returns:
        Tuple: Processed data and statistics.
    """
    preprocessing_parameters = preprocessing_parameters or PreprocessingParameters()
    data_columns = data_columns or DataColumns()

    try:
        date_range = initialize_dates(date_range=date_range)
        logging.info(f"Start date: {date_range.start_date}, End date: {date_range.end_date}")

        raw_df, total_uid, df_revenue_uid, total_revenue = fetch_data(
            data_fetch_params=data_fetch_params,
            date_range=date_range,
            preprocessing_parameters=preprocessing_parameters,
            data_columns=data_columns,
            csv_path=csv_path,
        )
        if raw_df is None:
            return None, None, None, None, None

        df_by_price = preprocess_data(raw_df=raw_df, data_columns=data_columns)

        return df_by_price, raw_df, total_uid, df_revenue_uid, total_revenue

    except ValidationError as e:
        logging.error(f"Validation error: {e}")
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None, None, None, None, None


def fetch_data(
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
    preprocessing_parameters: PreprocessingParameters,
    data_columns: DataColumns,
    csv_path: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[int], Optional[pd.DataFrame], Optional[int]]:
    """Handles data fetching and returns necessary components."""
    try:
        return progressive_monthly_aggregate(
            data_fetch_params=data_fetch_params,
            date_range=date_range,
            preprocessing_parameters=preprocessing_parameters,
            data_columns=data_columns,
            csv_path=csv_path,
        )
    except Exception as e:
        logging.error(f"Error retrieving data: {e}")
        return None, None, None, None


def preprocess_data(raw_df: pd.DataFrame, data_columns: DataColumns) -> pd.DataFrame:
    """Preprocesses the data after fetching."""
    try:
        df_by_price = preprocess_by_price(input_df=raw_df, data_columns=data_columns)
        last_values_df = calculate_last_values(input_df=raw_df, data_columns=data_columns)
        df_by_price = df_by_price.merge(last_values_df, on=[data_columns.uid], how="left")
        logging.info(f"Number of unique UIDs: {df_by_price[data_columns.uid].nunique()}")

        outliers_count = df_by_price[df_by_price["outlier_quantity"]][data_columns.uid].nunique()
        logging.info(f"Number of UIDs with outliers: {outliers_count}")

        return df_by_price

    except Exception as e:
        logging.error(f"Error during preprocessing by price: {e}")
        raise


def progressive_monthly_aggregate(
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
    preprocessing_parameters: Optional[PreprocessingParameters] = None,
    data_columns: Optional[DataColumns] = None,
    csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, int, pd.DataFrame, int]:
    """Perform progressive aggregation on monthly data.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_range (DateRange): The date range for fetching the data.
        preprocessing_parameters (Optional[PreprocessingParameters]): Preprocessing parameters.
        data_columns (Optional[DataColumns]): Configuration for data columns.
        csv_path (Optional[str]): Path to a CSV file. If provided, reads data from CSV instead.

    Returns:
        Tuple: Aggregated data and statistics.
    """
    preprocessing_parameters = preprocessing_parameters or PreprocessingParameters()
    data_columns = data_columns or DataColumns()

    approved_data_list, approved_uids = [], []
    dates_list = create_date_list(date_range=date_range)

    total_uid, total_revenue, df_revenue_uid = 0, 0, 0
    rejected_data = pd.DataFrame()

    for date_month in dates_list:
        _, rejected_data, approved_data = process_month_data(
            date_month=date_month,
            data_fetch_params=data_fetch_params,
            preprocessing_parameters=preprocessing_parameters,
            data_columns=data_columns,
            approved_uids=approved_uids,
            rejected_data=rejected_data,
            csv_path=csv_path,
        )
        approved_data_list.append(approved_data)

        if total_uid == 0:
            # Read data without filter to get all the uid
            df_revenue = read_data(
                data_fetch_params=data_fetch_params,
                data_month=date_month,
                data_columns=data_columns,
                filter_units=False,
                csv_path=csv_path,
            )
            total_uid, total_revenue, df_revenue_uid = get_revenue(
                df_revenue=df_revenue, data_columns=data_columns
            )

    log_rejection_reasons(
        rejected_data=rejected_data,
        preprocessing_parameters=preprocessing_parameters,
        data_columns=data_columns,
    )

    result_df = pd.concat(approved_data_list)
    result_df[data_columns.revenue] = result_df[data_columns.revenue].fillna(0).astype("float32")
    logging.info(f"Number of unique user IDs: {result_df[data_columns.uid].nunique()}")

    return result_df, total_uid, df_revenue_uid, total_revenue


def process_month_data(
    data_fetch_params: DataFetchParameters,
    date_month: str,
    preprocessing_parameters: PreprocessingParameters,
    data_columns: DataColumns,
    approved_uids: List[str],
    rejected_data: pd.DataFrame,
    csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Processes data for a single month, filters approved and rejected UIDs."""
    df_month = read_data(
        data_fetch_params=data_fetch_params,
        data_month=date_month,
        data_columns=data_columns,
        csv_path=csv_path,
    )

    data_to_test = pd.concat([df_month, rejected_data])
    data_to_test["outlier_quantity"] = data_to_test.groupby(data_columns.uid)[
        data_columns.quantity
    ].transform(outliers_iqr_filtered)

    uid_changes, uid_conversions = get_uid_changes_and_conversions(
        data_to_test=data_to_test,
        preprocessing_parameters=preprocessing_parameters,
        data_columns=data_columns,
    )
    approved_uids_month = list(set(uid_changes) & set(uid_conversions))
    approved_uids.extend(approved_uids_month)

    approved_data = data_to_test[data_to_test[data_columns.uid].isin(approved_uids_month)]
    rejected_data = data_to_test[~data_to_test[data_columns.uid].isin(approved_uids_month)]

    return df_month, rejected_data, approved_data


def read_data(
    data_fetch_params: DataFetchParameters,
    data_month: str,
    data_columns: DataColumns,
    filter_units: bool = True,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Read one month of data. Filter out inventory <= 0 and negative units. Optionally filter UIDs.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching such as
        client key, channel, and attribute name.
        data_month (str): The month of the data in "YYYY-MM-DD" format.
        data_columns (DataColumns): Configuration of data columns.
        filter_units (bool): True to get only rows where units > 0 (defaults to True).
        csv_path (Optional[str]): Path to a CSV file. If provided, reads data from CSV instead.

    Returns:
        pd.DataFrame: The DataFrame containing the filtered data.

    Raises:
        ValueError: If data_month is not a valid type.
        Exception: If there is an error reading the data.
    """
    data_month = parse_data_month(data_month)
    date_params = {
        "start_date": str(data_month.replace(day=1)),
        "end_date": str(
            data_month.replace(day=calendar.monthrange(data_month.year, data_month.month)[1])
        ),
    }

    try:
        if csv_path:
            df_read = pd.read_csv(csv_path, parse_dates=[data_columns.date])
            df_read = df_read[
                (df_read[data_columns.date] >= date_params["start_date"])
                & (df_read[data_columns.date] <= date_params["end_date"])
            ]
        else:
            df_read = read_data_query(
                data_fetch_params=data_fetch_params,
                date_params=date_params,
                filter_units=filter_units,
                data_columns=data_columns,
            )
        df_read = filter_data_by_uids(
            df=df_read, uids_to_filter=data_fetch_params.uids_to_filter, uid_column=data_columns.uid
        )
        df_read[data_columns.round_price] = df_read[data_columns.shelf_price].apply(
            round_price_effect
        )
        df_read = clean_inventory_data(df_input=df_read, inventory_column=data_columns.inventory)
        df_read = clean_quantity_data(df_input=df_read, quantity_column=data_columns.quantity)

    except Exception as e:
        logging.error(f"No data available for {data_month.strftime('%Y-%m')}: {e!s}")
        df_read = pd.DataFrame(
            columns=[
                data_columns.uid,
                data_columns.date,
                data_columns.shelf_price,
                data_columns.quantity,
                data_columns.revenue,
                data_columns.round_price,
            ]
        )

    return df_read
