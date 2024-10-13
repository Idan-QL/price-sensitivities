"""Module of report."""

import logging
from typing import Dict

import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from ql_toolkit.data_lake.athena_query import AthenaQuery


def get_previous_month_df(client_key: str, channel: str, end_date: str) -> pd.DataFrame:
    """Get the results from the last month.

    This function retrieves the results from the last month based on the
    provided end date, client key, and channel.

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        end_date (str): The end date.

    Returns:
        pd.DataFrame: The results from the last month as a pandas DataFrame.
    """
    # Compute the last day of the previous month
    previous_data_date = (pd.to_datetime(end_date).replace(day=1) - pd.DateOffset(days=1)).strftime(
        "%Y-%m-%d"
    )

    logging.info(
        f"Reading previous best_model from models_monitoring for client: {client_key}; "
        f"channel: {channel};"
        f"last_month_end_date: {previous_data_date};"
        f"end_date: {end_date} ..."
    )

    file_name = "elasticity_best_model"

    athena_query = AthenaQuery(
        client_key=client_key,
        channel=channel,
        file_name=file_name,
        previous_data_date=previous_data_date,
    )

    data_df = athena_query.execute_query()
    logging.info(
        f"Finishing reading previous best_model from models_monitoring. Shape: {data_df.shape}"
    )
    return data_df


def compare_model(df_results: pd.DataFrame, df_results_last_month: pd.DataFrame) -> int:
    """Compare the best model from the current month with the best model from the last month."""
    df_compare = df_results[["uid", "best_model", "quality_test"]].merge(
        df_results_last_month[["uid", "best_model"]],
        on="uid",
        how="left",
        suffixes=("", "_last_month"),
    )
    df_compare["model_changes"] = df_compare["best_model"] != df_compare["best_model_last_month"]
    return df_compare[df_compare["quality_test"]]["model_changes"].sum()


def get_elasticity_ranges_counts(
    df_filtered: pd.DataFrame, min_elasticity: float
) -> Dict[str, int]:
    """Calculate the count of UIDs for various elasticity ranges, ensuring integer counts.

    Raises an error if null values are present in the 'best_elasticity' column.

    Args:
        df_filtered (pd.DataFrame): The DataFrame containing filtered results.
        min_elasticity (float): The minimum elasticity threshold.

    Returns:
        Dict[str, int]: A dictionary mapping elasticity ranges to their corresponding counts.

    Raises:
        ValueError: If the 'best_elasticity' column contains null values.
    """
    if df_filtered["best_elasticity"].isna().any():
        raise ValueError("Null values found in 'best_elasticity' column.")
    return {
        "uids_below_minus6": int(len(df_filtered[df_filtered.best_elasticity < -6])),
        "uids_between_minus6_and_min": int(
            len(
                df_filtered[
                    (df_filtered.best_elasticity >= -6)
                    & (df_filtered.best_elasticity < min_elasticity)
                ]
            )
        ),
        "uids_between_min_and_minus1": int(
            len(
                df_filtered[
                    (df_filtered.best_elasticity >= min_elasticity)
                    & (df_filtered.best_elasticity < -1)
                ]
            )
        ),
        "uids_between_minus1_and_0": int(
            len(
                df_filtered[(df_filtered.best_elasticity >= -1) & (df_filtered.best_elasticity < 0)]
            )
        ),
        "uids_equal_0": int(len(df_filtered[df_filtered.best_elasticity == 0])),
        "uids_above_0": int(len(df_filtered[df_filtered.best_elasticity > 0])),
    }


def calculate_percentage_metrics(
    df_valid_elasticity: pd.DataFrame,
    total_uid: int,
    total_revenue: int,
    uids_with_elasticity_data: int,
    total_revenue_with_elasticity_data: int,
) -> Dict[str, float]:
    """Calculate various percentage metrics based on UIDs and revenue.

    Args:
        df_valid_elasticity (pd.DataFrame): The DataFrame containing filtered results.
        total_uid (int): The total number of UIDs in the dataset.
        total_revenue (int): The total revenue from all UIDs.
        uids_with_elasticity_data (int): The count of unique UIDs in the results.
        total_revenue_with_elasticity_data (int): The total revenue from UIDs with data
        for elasticity.

    Returns:
        Dict[str, float]: A dictionary containing percentage-based metrics for UIDs and revenue.
        If total_revenue or total_revenue_with_elasticity_data is non-positive, all percentages
        are set to 0 and an error is logged.
    """
    if total_revenue <= 0 or total_revenue_with_elasticity_data <= 0:
        logging.error(
            "Invalid revenue values: total_revenue=%s, total_revenue_with_elasticity_data=%s. "
            "Metrics will return 0.",
            total_revenue,
            total_revenue_with_elasticity_data,
        )
        return {
            "percentage_uids_from_total": 0.0,
            "percentage_revenue_from_total": 0.0,
            "percentage_uids_with_elasticity_data": 0.0,
            "percentage_revenue_with_elasticity_data": 0.0,
        }

    return {
        "percentage_uids_from_total": round(len(df_valid_elasticity) / total_uid * 100, 1),
        "percentage_revenue_from_total": round(
            df_valid_elasticity["revenue"].sum() / total_revenue * 100, 1
        ),
        "percentage_uids_with_elasticity_data": round(
            len(df_valid_elasticity) / uids_with_elasticity_data * 100, 1
        ),
        "percentage_revenue_with_elasticity_data": round(
            df_valid_elasticity["revenue"].sum() / total_revenue_with_elasticity_data * 100,
            1,
        ),
    }


def get_model_type_counts(df_filtered: pd.DataFrame) -> Dict[str, int]:
    """Retrieve the count of UIDs for different model types.

    Args:
        df_filtered (pd.DataFrame): The DataFrame containing filtered results.

    Returns:
        Dict[str, int]: A dictionary mapping model types to their corresponding counts.
    """
    model_counts = df_filtered["best_model"].value_counts()
    return {
        "power_model_count": model_counts.get("power", 0),
        "exponential_model_count": model_counts.get("exponential", 0),
        "linear_model_count": model_counts.get("linear", 0),
    }


def get_elasticity_type_counts(df_filtered: pd.DataFrame) -> Dict[str, int]:
    """Retrieve the count of UIDs for different elasticity types.

    Args:
        df_filtered (pd.DataFrame): The DataFrame containing filtered results.

    Returns:
        Dict[str, int]: A dictionary mapping elasticity types to their corresponding counts.
    """
    type_counts = df_filtered["type"].value_counts()
    return {
        "uid_type_count": type_counts.get("uid", 0),
        "group1_type_count": type_counts.get("group 1", 0),
        "group2_type_count": type_counts.get("group 2", 0),
    }


def generate_run_report(
    data_fetch_params: DataFetchParameters,
    total_uid: int,
    results_df: pd.DataFrame,
    runtime_duration: float,
    total_revenue: int,
    error_count: int,
    end_date: str,
    is_qa_run: bool,
    min_elasticity: float = -3.8,
) -> pd.DataFrame:
    """Generate a report DataFrame for the current run, ensuring appropriate types for each field.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        total_uid (int): The total number of UIDs in the run.
        results_df (pd.DataFrame): The DataFrame containing the run results.
        runtime_duration (float): The runtime duration in seconds.
        total_revenue (int): The total revenue for the client.
        error_count (int): The number of errors encountered during the run.
        end_date (str): The date marking the end of the run period.
        is_qa_run (bool): Save is_qa_run value.
        min_elasticity (float, optional): The minimum elasticity threshold. Defaults to -3.8.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the calculated metrics for the run.
    """
    uids_with_elasticity_data = results_df.uid.nunique()
    total_revenue_with_data = results_df[results_df.type == "uid"].revenue.sum()

    # Filter results based on conditions
    df_valid_elasticity = results_df[results_df["quality_test"] & results_df["result_to_push"]]
    high_quality_results = results_df[
        results_df["quality_test_high"] & results_df["result_to_push"]
    ]

    # Handle model changes if applicable
    model_changes = None
    previous_month_results = get_previous_month_df(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        end_date=end_date,
    )
    if previous_month_results is not None and not previous_month_results.empty:
        model_changes = compare_model(results_df, previous_month_results)

    # Calculate various metrics and ensure correct types
    elasticity_counts = get_elasticity_ranges_counts(df_valid_elasticity, min_elasticity)
    percentage_metrics = calculate_percentage_metrics(
        df_valid_elasticity,
        total_uid,
        total_revenue,
        uids_with_elasticity_data,
        total_revenue_with_data,
    )
    model_type_counts = get_model_type_counts(df_valid_elasticity)
    elasticity_type_counts = get_elasticity_type_counts(df_valid_elasticity)

    # Prepare the report row
    report_row = {
        "client_key": data_fetch_params.client_key,
        "channel": data_fetch_params.channel,
        "total_uid": total_uid,
        "uids_with_elasticity": len(df_valid_elasticity),
        "uids_with_high_quality_elasticity": len(high_quality_results),
        **percentage_metrics,
        "uids_with_elasticity_data": uids_with_elasticity_data,
        **elasticity_counts,
        **elasticity_type_counts,
        **model_type_counts,
        "runtime_duration": runtime_duration,
        "model_changes": model_changes,
        "error_count": error_count,
    }

    # Add is_qa_run only if it is True
    report_row["run_type"] = "QA" if is_qa_run else "Production"

    # Convert to DataFrame and ensure correct types for numeric columns
    df_report = pd.DataFrame([report_row])

    # Cast specific columns to their correct types, ensuring float32 for floats and int for integers
    return df_report.astype(
        {
            "total_uid": "int32",
            "uids_with_elasticity": "int32",
            "uids_with_high_quality_elasticity": "int32",
            "uids_with_elasticity_data": "int32",
            "runtime_duration": "float32",
            "error_count": "int32",
        }
    )
