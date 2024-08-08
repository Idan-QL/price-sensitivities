"""Module of group modeling."""

import logging
from typing import List, Optional

import pandas as pd

import elasticity.model.run_model as run_model


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


def run_group_experiment(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Run the model experiment for the given DataFrame and group column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        group_col (str): The column name to group by.

    Returns:
        pd.DataFrame: The result DataFrame from running the experiment.
    """
    return run_model.run_experiment_for_uids_parallel(
        df_input=df,
        uid_col=group_col,
        price_col="round_price",
        quantity_col="units",
        weights_col="days",
    )


def merge_with_original_uids(
    df_results: pd.DataFrame, df_group: pd.DataFrame, group_col: str
) -> pd.DataFrame:
    """Merge the results DataFrame with the original DataFrame to get the original UIDs.

    Args:
        df_results (pd.DataFrame): The DataFrame containing the results.
        df_group (pd.DataFrame): The original DataFrame containing the group data.
        group_col (str): The column name to group by.

    Returns:
        pd.DataFrame: The merged DataFrame with original UIDs.
    """
    uid_group_uid_df = df_group.drop_duplicates(subset=["uid", group_col])[["uid", group_col]]
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


def process_group_segmentation(df_group: pd.DataFrame, segmentation_column: str) -> pd.DataFrame:
    """Process a specific group segmentation.

    Args:
        df_group (pd.DataFrame): The original DataFrame containing the group data.
        segmentation_column (str): The column name for the group segmentation.

    Returns:
        pd.DataFrame: The results of the group experiment merged with original UIDs.
    """
    group_uids = get_group_uids(df_group, segmentation_column)
    df_by_price_group = df_group[df_group[segmentation_column].isin(group_uids)]
    df_results_group = run_group_experiment(df_by_price_group, segmentation_column)
    return merge_with_original_uids(df_results_group, df_group, segmentation_column)


def add_group_elasticity(df_group: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    """Process the data to run elasticity on group 1 and 2.

    Args:
        df_group (pd.DataFrame): The original DataFrame containing the group data.
        df_results (pd.DataFrame): The DataFrame containing the overall results.

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
    df_results_group_1 = process_group_segmentation(df_group, "group_uid_segmentation_1")
    df_results_group_2 = process_group_segmentation(df_group, "group_uid_segmentation_2")

    # Add group_type column and concatenate the results
    df_results_group_1["type"] = "group 1"
    df_results_group_2["type"] = "group 2"
    df_results["type"] = "uid"
    df_results["result_to_push"] = df_results["quality_test"]

    # Set the result_to_push flag
    set_result_flag(df_results_group_1, df_results)
    set_result_flag(df_results_group_2, df_results, df_results_group_1)

    return pd.concat([df_results, df_results_group_1, df_results_group_2], ignore_index=True)
