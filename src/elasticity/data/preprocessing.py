"""Module of preprocessing."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from elasticity.data.utils import (
    preprocess_by_price,
    round_price_effect,
    uid_with_min_conversions,
    uid_with_price_changes,
    outliers_iqr_filtered
)
from ql_toolkit.config.runtime_config import app_state


def read_and_preprocess(
    client_key: str,
    channel: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bucket: Optional[str] = None,
    uids_to_filter: Optional[str] = None,
    dir_: str = "data_science/datasets",
    price_changes: int = 5,
    threshold: float = 0.01,
    min_days_with_conversions: int = 10,
    uid_col: str = "uid",
    price_col: str = "round_price",
    quantity_col: str = "units",
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and preprocess the DataFrame.

    TODO: add original price_col base for round_price.

    Returns:
    - df_by_day (DataFrame): DataFrame grouped by day.
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=2)).replace(day=1) - relativedelta(
            months=1
        )
        end_date = end_date.strftime("%Y-%m-%d")

    if start_date is None:
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_date_dt = end_date_dt - relativedelta(months=11)
        start_date = start_date_dt.strftime("%Y-%m-%d")

    if bucket is None:
        bucket = app_state.bucket_name

    logging.info(f"start_date: {start_date}")
    logging.info(f"end_date: {end_date}")

    df, total_uid, df_revenue_uid, total_revenue = progressive_monthly_aggregate(
        client_key=client_key,
        channel=channel,
        bucket=bucket,
        start_date=start_date,
        end_date=end_date,
        uids_to_filter=uids_to_filter,
        dir_=dir_,
        price_changes=price_changes,
        threshold=threshold,
        min_days_with_conversions=min_days_with_conversions,
        uid_col=uid_col,
        price_col=price_col,
        quantity_col=quantity_col,
    )

    df_by_price = preprocess_by_price(
        df,
        uid_col=uid_col,
        date_col=date_col,
        price_col=price_col,
        quantity_col=quantity_col,
    )

    return df_by_price, df, total_uid, end_date, df_revenue_uid, total_revenue


def progressive_monthly_aggregate(
    client_key: str,
    channel: str,
    bucket: str,
    start_date: str,
    end_date: str,
    uids_to_filter: Optional[str] = None,
    dir_: str = "data_science/datasets",
    price_changes: int = 4,
    threshold: float = 0.01,
    min_days_with_conversions: int = 15,
    uid_col: str = "uid",
    price_col: str = "round_price",
    quantity_col: str = "units",
) -> pd.DataFrame:
    """Read monthly data and concatenate into a single DataFrame."""
    df_full_list = []
    uid_ok = []
    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    df_ko = pd.DataFrame()  # Initialize df_ko
    total_uid = 0
    while end_date_dt >= start_date_dt:
        logging.info(f'reading: {end_date_dt.strftime("%Y-%m-%d")}')
        df_part = read_data(
            client_key,
            channel,
            bucket,
            uids_to_filter=uids_to_filter,
            date=end_date_dt.strftime("%Y-%m-%d"),
            dir_=dir_,
        )

        # delete uid where inventory is 0
        logging.info(
            f'Number of inventory less or equal to 0: {len(df_part[df_part["inventory"] <= 0])}'
        )
        df_part = df_part[df_part["inventory"] > 0]
        del df_part["inventory"]
        # Process data
        df_part = process_data(df_part)

        if total_uid == 0:
            df_revenue = df_part.copy()
            df_revenue["revenue"] = df_revenue["price_merged"] * df_revenue["units"]
            total_revenue = df_revenue["revenue"].sum()
            df_revenue_uid = df_revenue.groupby(["uid"])["revenue"].sum().reset_index()
            df_revenue_uid["revenue_percentage"] = (
                df_revenue_uid["revenue"] / total_revenue
            )

            total_uid = df_part["uid"].nunique()
            logging.info(f"Total uid: {total_uid}")
        df_part = df_part[~df_part["uid"].isin(uid_ok)]

        # Concatenate df_part with df_ko
        df_part = pd.concat([df_part, df_ko])

        df_part['outlier_quantity'] = df_part.groupby(uid_col)[quantity_col].transform(
        outliers_iqr_filtered)

        uid_changes = uid_with_price_changes(
                df_part,
                price_changes=price_changes,
                threshold=threshold,
                price_col=price_col,
                quantity_col=quantity_col)

        uid_conversions = uid_with_min_conversions(
            df_part,
            min_days_with_conversions=min_days_with_conversions,
            uid_col=uid_col,
            quantity_col=quantity_col,
        )
        uid_intersection_change_conversions = list(
            set(uid_changes) & set(uid_conversions)
        )
        uid_ok.extend(uid_intersection_change_conversions)

        df_ok = df_part[df_part["uid"].isin(uid_intersection_change_conversions)]
        logging.info(f"number of uid ok: {len(uid_ok)}")
        df_ko = df_part[~df_part["uid"].isin(uid_intersection_change_conversions)]

        df_full_list.append(df_ok)
        end_date_dt -= pd.DateOffset(months=1)

    result_df = pd.concat(df_full_list)
    logging.info(f"Number of unique user IDs: {result_df.uid.nunique()}")
    return result_df, total_uid, df_revenue_uid, total_revenue


def read_data(
    client_key: str,
    channel: str,
    bucket: str,
    uids_to_filter: Optional[str] = None,
    date: str = "2024-02-01",
    dir_: str = "data_science/datasets",
) -> pd.DataFrame:
    """Read monthly data and concatenate into a single DataFrame."""
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
    try:
        filters = (
            [("uid", "in", uids_to_filter)] if uids_to_filter is not None else None
        )
        df_read = pd.read_parquet(
            f"s3://{bucket}/{dir_}/{client_key}/{channel}/elasticity/{year_}_{int(month_)}_full_data.parquet/",
            columns=cs,
            filters=filters,
        )
    except Exception:
        logging.error(f"No data for {year_!s}_{int(month_)!s}")
        logging.info(
            f"s3://{bucket}/{dir_}/{client_key}/{channel}/elasticity/{year_}_{int(month_)}_full_data.parquet/"
        )
        df_read = pd.DataFrame(columns=cs)
        pass
    return df_read


def process_data(df_full: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional columns based on the read monthly data.

    if conversion price is available, use conversion price,
    else if view price is avaialable, use views price
    else use price recommendations
    return price round
    """
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

    df_full["units"] = np.where(~df_full.total_units.isna(), df_full.total_units, 0)
    df_full["round_price"] = df_full["price_merged"].apply(round_price_effect)

    return df_full[["date", "uid", "round_price", "units", "price_merged", "source"]]
