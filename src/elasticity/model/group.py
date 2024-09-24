"""Module of group modeling."""

import copy
import logging
from typing import List, Optional

import pandas as pd

import elasticity.model.run_model as run_model
from elasticity.data.configurator import DataColumns, DataFetchParameters, DateRange
from elasticity.data.group import data_for_group_elasticity


def handle_group_elasticity(
    df_by_price: pd.DataFrame,
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
    df_results: pd.DataFrame,
    data_columns: DataColumns,
) -> pd.DataFrame:
    """Handle group elasticity if the attribute is specified.

    Args:
        df_by_price (pd.DataFrame): The DataFrame containing price data.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching
        such as client key, channel, and attribute name.
        date_range (DateRange): The date range for fetching the data.
        df_results (pd.DataFrame): The DataFrame containing experiment results.
        data_columns (DataColumns): Configuration for column mappings used in the experiment.

    Returns:
        pd.DataFrame: The updated DataFrame with group elasticity information.
    """
    if data_fetch_params.attr_name:
        logging.info(f"Running group elasticity - attr: {data_fetch_params.attr_name}")
        df_group = data_for_group_elasticity(
            df_by_price=df_by_price,
            data_fetch_params=data_fetch_params,
            end_date=date_range.end_date,
        )
        return add_group_elasticity(
            df_group=df_group, df_results=df_results, data_columns=data_columns
        )

    logging.info(f"Skipping group elasticity - attr: {data_fetch_params.attr_name}")
    df_results["result_to_push"] = df_results["quality_test"]
    df_results["type"] = "uid"
    return df_results


def get_group_uids(df: pd.DataFrame, group_col: str) -> List[str]:
    """Filter out outliers and find group UIDs with more than one unique UID.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The column name to group by.

    Returns:
        List[str]: A list of group UIDs with more than one unique UID.
    """
    return (
        df[~df["outlier_quantity"]]
        .groupby(group_col)["uid"]
        .nunique()
        .loc[lambda x: x > 1]
        .index.tolist()
    )


def run_group_experiment(
    df_input: pd.DataFrame, data_columns: DataColumns, group_col: str
) -> pd.DataFrame:
    """Run the model experiment for the given DataFrame and group column.

    Args:
        df_input (pd.DataFrame): The DataFrame containing the data.
        data_columns (DataColumns): Configuration of data columns.
        group_col (str): The column name to group by.

    Returns:
        pd.DataFrame: The result DataFrame from running the experiment.
    """
    data_columns_group = copy.deepcopy(data_columns)
    data_columns_group.uid = group_col

    return run_model.run_experiment_for_uids_parallel(
        df_input=df_input, data_columns=data_columns_group
    )


def merge_with_original_uids(
    df_results: pd.DataFrame, df_group: pd.DataFrame, group_col: str, data_columns: DataColumns
) -> pd.DataFrame:
    """Merge the results DataFrame with the original DataFrame to get the original UIDs.

    Args:
        df_results (pd.DataFrame): The DataFrame containing the results.
        df_group (pd.DataFrame): The original DataFrame containing the group data.
        group_col (str): The column name to group by.
        data_columns (DataColumns): Configuration of data columns.

    Returns:
        pd.DataFrame: The merged DataFrame with original UIDs.
    """
    uid_group_uid_df = df_group.drop_duplicates(subset=[data_columns.uid, group_col])[
        [data_columns.uid, group_col]
    ]
    return df_results.merge(uid_group_uid_df, on=group_col)


def set_result_flag(
    df_results_group: pd.DataFrame,
    df_results: pd.DataFrame,
    other_df: Optional[pd.DataFrame] = None,
) -> None:
    """Set the result_to_push flag with additional condition and update detail column.

    Args:
        df_results_group (pd.DataFrame): The DataFrame containing the group results.
        df_results (pd.DataFrame): The DataFrame containing the overall results.
        other_df (Optional[pd.DataFrame]): Another DataFrame for additional checking.

    Returns:
        None
    """
    condition = (
        ~df_results_group["uid"].isin(df_results[df_results["quality_test"]]["uid"])
    ) & df_results_group["quality_test"]
    if other_df is not None:
        condition &= ~df_results_group["uid"].isin(other_df[other_df["quality_test"]]["uid"])
    df_results_group["result_to_push"] = condition

    elasticity_label = f"Elasticity {df_results_group['type'].iloc[0]}"
    df_results_group.loc[condition, "details"] += f" | {elasticity_label}"


def process_group_segmentation(
    df_group: pd.DataFrame, segmentation_column: str, data_columns: DataColumns
) -> pd.DataFrame:
    """Process a specific group segmentation.

    Args:
        df_group (pd.DataFrame): The original DataFrame containing the group data.
        segmentation_column (str): The column name for the group segmentation.
        data_columns (DataColumns): Configuration of data columns.

    Returns:
        pd.DataFrame: The results of the group experiment merged with original UIDs.
    """
    group_uids = get_group_uids(df_group, segmentation_column)
    df_by_price_group = df_group[df_group[segmentation_column].isin(group_uids)]
    df_results_group = run_group_experiment(
        df_input=df_by_price_group, data_columns=data_columns, group_col=segmentation_column
    )

    return merge_with_original_uids(
        df_results=df_results_group,
        df_group=df_group,
        group_col=segmentation_column,
        data_columns=data_columns,
    )


def add_group_elasticity(
    df_group: pd.DataFrame, df_results: pd.DataFrame, data_columns: DataColumns
) -> pd.DataFrame:
    """Process the data to run elasticity on group 1 and 2.

    Args:
        df_group (pd.DataFrame): The original DataFrame containing the group data.
        df_results (pd.DataFrame): The DataFrame containing the overall results.
        data_columns (DataColumns): Configuration of data columns.

    Returns:
        pd.DataFrame: The combined DataFrame with processed results. If `df_group`
        is empty, logs an error and returns `df_results`.
    """
    if df_group.empty:
        logging.error("No group elasticity data available.")
        df_results["result_to_push"] = df_results["quality_test"]
        df_results["type"] = "uid"
        return df_results

    # Process both group segmentations
    df_results_group_1 = process_group_segmentation(
        df_group=df_group, segmentation_column="group_uid_segmentation_1", data_columns=data_columns
    )
    df_results_group_2 = process_group_segmentation(
        df_group=df_group, segmentation_column="group_uid_segmentation_2", data_columns=data_columns
    )

    # Add group_type column and concatenate the results
    df_results_group_1["type"] = "group 1"
    df_results_group_2["type"] = "group 2"
    df_results["type"] = "uid"
    df_results["result_to_push"] = df_results["quality_test"]

    # Set the result_to_push flag
    set_result_flag(df_results_group_1, df_results)
    set_result_flag(df_results_group_2, df_results, df_results_group_1)

    return pd.concat([df_results, df_results_group_1, df_results_group_2], ignore_index=True)
