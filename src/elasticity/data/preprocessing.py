"""Module of preprocessing."""

import calendar
import logging
from datetime import date, datetime
from typing import Optional, Tuple

import pandas as pd

from elasticity.data.configurator import DataColumns, PreprocessingParameters
from elasticity.data.utils import (
    calculate_last_values,
    get_revenue,
    initialize_dates,
    outliers_iqr_filtered,
    preprocess_by_price,
    round_price_effect,
    uid_with_min_conversions,
    uid_with_price_changes,
)
from ql_toolkit.s3.pya import query_agg_day_client_data


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
            raw_df, total_uid, df_revenue_uid, total_revenue = (
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
            # Add the last price and date by uid
            last_values_df = calculate_last_values(raw_df, data_columns)
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return None, None, None, None, None

        # Data preprocessing
        try:
            df_by_price = preprocess_by_price(raw_df, data_columns=data_columns)
            # add last price and date
            df_by_price = df_by_price.merge(
                last_values_df, on=[data_columns.uid], how="left"
            )
            logging.info(f"Number of unique UID : {df_by_price.uid.nunique()}")
            # df_by_price = df_by_price[df_by_price[data_columns.quantity] >
            # preprocessing_parameters.avg_daily_sales]
            # uid_ok = (df_by_price.groupby('uid').units.count()[df_by_price.groupby('uid').
            # units.count()>5]).index.tolist()
            # df_by_price = df_by_price[df_by_price.uid.isin(uid_ok)]
            # logging.info(f"Number of unique UID after filter avg_daily_sales :
            # {df_by_price.uid.nunique()}")
            outliers_count = df_by_price[df_by_price["outlier_quantity"]].uid.nunique()
            logging.info(f"Number of uid with outliers: {outliers_count}")
        except Exception as e:
            logging.error(f"Error during preprocessing by price: {e}")
            return None, raw_df, total_uid, df_revenue_uid, total_revenue

        logging.info(f"start_date: {start_date}")
        logging.info(f"end_date: {end_date}")
        logging.info(f"Total number of uid: {total_uid}")
        logging.info(f"total_revenue: {total_revenue}")

        return df_by_price, raw_df, total_uid, df_revenue_uid, total_revenue

    except ValueError as e:
        logging.error(f"ValueError: {e}")
        return None, None, None, None, None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None, None, None, None, None


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
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    dates_list = pd.date_range(start=start_date_dt, end=end_date_dt, freq="ME").strftime(
        "%Y-%m-%d"
    )[::-1]

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
            date_read=date_month,
        )

        df_month[data_columns.price] = df_month["shelf_price"].apply(round_price_effect)

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
        # logging.info(f"number of uid ok: {len(approved_uids)}")
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
    date_read: str = "2024-02-01",
) -> pd.DataFrame:
    """Read one month data. Filter out inventory 0 and negative unit. Option to filter one uids.

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        uids_to_filter (Optional[str], optional): A list of uids to filter the data.
        Defaults to None.
        date_read (str, optional): The date in the format "YYYY-MM-DD".
        Defaults to "2024-02-01".

    Returns:
        pd.DataFrame: The DataFrame containing the read data.

    Raises:
        Exception: If there is an error reading the data.

    """
    if isinstance(date_read, datetime):
        date_read = date_read.date()
    elif isinstance(date_read, str):
        date_read = datetime.strptime(date_read, "%Y-%m-%d").date()
    elif not isinstance(date_read, date):
        raise ValueError("Input must be a string, date, or datetime object")
    cs = [
        "date",
        "uid",
        "shelf_price",
        "units",
        "price",
        "inventory",
    ]

    date_params = {
        "start_date": str(date_read.replace(day=1)),
        "end_date": str(
            date_read.replace(
                day=calendar.monthrange(date_read.year, date_read.month)[1]
            )
        ),
    }

    try:
        df_read = query_agg_day_client_data(client_key, channel, date_params, True)
        # filter out uids already complete
        if uids_to_filter is not None:
            df_read = df_read[~df_read.uid.isin(uids_to_filter)]

        logging.info(
            f'Number of inventory less or equal to 0: {len(df_read[df_read["inventory"] <= 0])}'
        )
        df_read = df_read[
            (df_read["inventory"] > 0) | (df_read["inventory"].isna())
        ].drop(columns=["inventory"])

        logging.info(f'Number of negative unit: {len(df_read[df_read["units"] < 0])}')
        df_read["units"] = df_read["units"].fillna(0)
        df_read = df_read[df_read["units"] >= 0]
    except Exception:
        year_, month_ = date.strftime("%Y-%m").split("-")
        logging.error(f"No data for {year_!s}_{int(month_)!s}")
        df_read = pd.DataFrame(columns=cs)
        pass
    return df_read
