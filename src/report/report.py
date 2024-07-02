"""Module of report."""

import logging
import traceback
from typing import List

import pandas as pd

from ql_toolkit.s3 import io as s3io


def compare_model(df_results: pd.DataFrame, df_results_last_month: pd.DataFrame) -> int:
    """Compare the best model from the current month with the best model from the last month."""
    df_compare = df_results[["uid", "best_model", "quality_test"]].merge(
        df_results_last_month[["uid", "best_model"]],
        on="uid",
        how="left",
        suffixes=("", "_last_month"),
    )
    df_compare["model_changes"] = (
        df_compare["best_model"] != df_compare["best_model_last_month"]
    )
    return df_compare[df_compare["quality_test"]]["model_changes"].sum()


def get_last_month_results(
    client_key: str, channel: str, end_date: str
) -> pd.DataFrame:
    """Get the results from the last month.

    This function retrieves the results from the last month based on the
    provided end date, client key, and channel.
    # TODO: change only to last month in one month as
    # we just changed to save the file as end of the month

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        end_date (str): The end date.

    Returns:
        pd.DataFrame: The results from the last month as a pandas DataFrame.
    """
    # Compute the last day of the previous month in one line
    last_month_end_date = (
        pd.to_datetime(end_date).replace(day=1) - pd.DateOffset(days=1)
    ).strftime("%Y-%m-%d")

    # Try reading the results from the last day of the previous month
    try:
        df_results_last_month = s3io.maybe_get_pd_csv_df(
            file_name=f"elasticity_{client_key}_{channel}_{last_month_end_date}.csv",
            s3_dir="data_science/eval_results/elasticity/",
        )
    except Exception:
        logging.error(traceback.format_exc())
        logging.error(
            "No results found for last month (end date) %s - %s - %s",
            client_key,
            channel,
            last_month_end_date,
        )

    # Try reading the results from the first day of the previous month
    try:
        # Compute the first day of the previous month
        first_day_previous_month = (
            (pd.to_datetime(end_date) - pd.DateOffset(months=1))
            .replace(day=1)
            .strftime("%Y-%m-%d")
        )

        df_results_last_month = s3io.maybe_get_pd_csv_df(
            file_name=f"elasticity_{client_key}_{channel}_{first_day_previous_month}.csv",
            s3_dir="data_science/eval_results/elasticity/",
        )
    except Exception:
        logging.error(traceback.format_exc())
        logging.error(
            "No results found for last month (start date) %s - %s - %s",
            client_key,
            channel,
            first_day_previous_month,
        )
        # Return an empty DF
        df_results_last_month = pd.DataFrame()

    return df_results_last_month


def add_run(
    data_report: List,
    client_key: str,
    channel: str,
    total_uid: int,
    df_results: pd.DataFrame,
    runtime: float,
    total_revenue: int,
    error_counter: int,
    end_date: str,
    max_elasticity: float = 3.8,
    min_elasticity: float = -3.8,
) -> None:
    """Append data to the data report list.

    Args:
        data_report (List): The list to append data to.
        client_key (str): The client key.
        channel (str): The channel.
        total_uid (int): The total UID.
        df_results (pd.DataFrame): The DataFrame containing results.
        runtime (float): The runtime duration.
        total_revenue (int): The total revenue.
        error_counter (int): The error counter.
        end_date (str): The end date.
        max_elasticity (float, optional): The maximum elasticity value. Defaults to 3.8.
        min_elasticity (float, optional): The minimum elasticity value. Defaults to -3.8.

    Returns:
        List
    """
    uid_results_count = df_results.uid.nunique()
    df_results_quality = df_results[
        df_results["quality_test"] & df_results["result_to_push"]
    ]
    df_results_quality_high = df_results[
        df_results["quality_test_high"] & df_results["result_to_push"]
    ]
    best_model_counts = df_results_quality["best_model"].value_counts()
    elasticity_type_counts = df_results_quality["type"].value_counts()

    model_changes = None
    df_lastmonth = get_last_month_results(client_key, channel, end_date)
    if not df_lastmonth.empty:
        model_changes = compare_model(df_results, df_lastmonth)

    data_report.append(
        {
            "client_key": client_key,
            "channel": channel,
            "total_uid": total_uid,
            "uid_with_elasticity": len(df_results_quality),
            "uid_with_elasticity_high_quality": len(df_results_quality_high),
            "uids_from_total": round(len(df_results_quality) / total_uid * 100, 1),
            "revenue_from_total": round(
                df_results_quality["revenue"].sum() / total_revenue * 100, 1
            ),
            "uids_from_total_with_data": round(
                len(df_results_quality) / uid_results_count * 100, 1
            ),
            "revenue_from_total_with_data": round(
                (df_results_quality["revenue"].sum() /
                 df_results[df_results.type == 'uid'].revenue.sum() * 100),
                1,
            ),
            "uid_with_data_for_elasticity": uid_results_count,
            "uid_with_elasticity_less_than_minus6": len(
                df_results_quality[df_results_quality.best_elasticity < -6]
            ),
            "uid_with_elasticity_moreorequal_minus6_less_than_minus3.8": len(
                df_results_quality[
                    (df_results_quality.best_elasticity >= -6)
                    & (df_results_quality.best_elasticity < min_elasticity)
                ]
            ),
            "uid_with_elasticity_moreorequal_minus3.8_less_than_minus1": len(
                df_results_quality[
                    (df_results_quality.best_elasticity >= min_elasticity)
                    & (df_results_quality.best_elasticity < -1)
                ]
            ),
            "uid_with_elasticity_moreorequal_minus1_less_than_0": len(
                df_results_quality[
                    (df_results_quality.best_elasticity >= -1)
                    & (df_results_quality.best_elasticity < 0)
                ]
            ),
            "uid_with_elasticity_moreorequal_0_less_than_1": len(
                df_results_quality[
                    (df_results_quality.best_elasticity >= 0)
                    & (df_results_quality.best_elasticity < 1)
                ]
            ),
            "uid_with_elasticity_moreorequal_1_less_than_3.8": len(
                df_results_quality[
                    (df_results_quality.best_elasticity >= 1)
                    & (df_results_quality.best_elasticity < max_elasticity)
                ]
            ),
            "uid_with_elasticity_moreorequal_3.8_less_than_6": len(
                df_results_quality[
                    (df_results_quality.best_elasticity >= max_elasticity)
                    & (df_results_quality.best_elasticity < 6)
                ]
            ),
            "uid_with_elasticity_moreorequal_6": len(
                df_results_quality[df_results_quality.best_elasticity >= 6]
            ),
            "type_uid": elasticity_type_counts.get("uid", 0),
            "type_group1": elasticity_type_counts.get("group 1", 0),
            "type_group2": elasticity_type_counts.get("group 2", 0),
            "best_model_power_count": best_model_counts.get("power", 0),
            "best_model_exponential_count": best_model_counts.get("exponential", 0),
            "best_model_linear_count": best_model_counts.get("linear", 0),
            "runtime_duration": runtime,
            "model_changes": model_changes,
            "error": error_counter,
        }
    )

    return data_report


def add_error_run(
    data_report: List, client_key: str, channel: str, error_counter: int
) -> None:
    """Append data to the data report list.

    Args:
        data_report (List): The list to append data to.
        client_key (str): The client key.
        channel (str): The channel.
        error_counter (int): The counter for errors.

    Returns:
        None
    """
    data_report.append(
        {
            "client_key": client_key,
            "channel": channel,
            "total_uid": None,
            "uid_with_elasticity": None,
            "uids_from_total": None,
            "revenue_from_total": None,
            "uids_from_total_with_data": None,
            "revenue_from_total_with_data": None,
            "uid_with_data_for_elasticity": None,
            "uid_with_elasticity_less_than_minus6": None,
            "uid_with_elasticity_moreorequal_minus6_less_than_minus3.8": None,
            "uid_with_elasticity_moreorequal_minus3.8_less_than_minus1": None,
            "uid_with_elasticity_moreorequal_minus1_less_than_0": None,
            "uid_with_elasticity_moreorequal_0_less_than_1": None,
            "uid_with_elasticity_moreorequal_1_less_than_3.8": None,
            "uid_with_elasticity_moreorequal_3.8_less_than_6": None,
            "uid_with_elasticity_moreorequal_6": None,
            "best_model_power_count": None,
            "best_model_exponential_count": None,
            "best_model_linear_count": None,
            "runtime_duration": None,
            "model_changes": None,
            "error": error_counter,
        }
    )
