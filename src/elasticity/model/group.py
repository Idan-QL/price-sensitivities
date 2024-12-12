"""Module of group modeling."""

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import elasticity.model.run_model as run_model
from elasticity.data.configurator import DataColumns, DataFetchParameters, DateRange
from elasticity.data.group import data_for_group_elasticity
from elasticity.utils.consts import OUTPUT_CS


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
    if data_fetch_params.attr_names:
        logging.info(f"Running group elasticity - attr: {data_fetch_params.attr_names}")
        df_group = data_for_group_elasticity(
            df_by_price=df_by_price,
            data_fetch_params=data_fetch_params,
            end_date=date_range.end_date,
        )
        # TODO: For xxxls only
        # df_group.to_csv("df_group.csv", index=False)
        # TODO: Test group on all and uncomment if approved
        # df_results = add_missing_uids(df_results=df_results, df_by_price_all=df_by_price)

        return add_group_elasticity(
            df_group=df_group, df_results=df_results, data_columns=data_columns
        )

    logging.info(f"Skipping group elasticity - attr: {data_fetch_params.attr_names}")
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
) -> pd.DataFrame:
    """Set the result_to_push flag with additional condition and update detail column.

    Args:
        df_results_group (pd.DataFrame): The DataFrame containing the group results.
        df_results (pd.DataFrame): The DataFrame containing the overall results.
        other_df (Optional[pd.DataFrame]): Another DataFrame for additional checking.

    Returns:
        pd.DataFrame: A new DataFrame with updated result_to_push and details columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_updated = df_results_group.copy()

    # Define the initial condition
    condition = (
        ~df_updated["uid"].isin(df_results[df_results["quality_test"]]["uid"])
    ) & df_updated["quality_test"]

    # Apply additional condition if other_df is provided
    if other_df is not None:
        condition &= ~df_updated["uid"].isin(other_df[other_df["quality_test"]]["uid"])

    # Set the result_to_push flag based on the condition
    df_updated["result_to_push"] = condition

    # Update the details column where the condition is True
    if not df_updated.empty:
        elasticity_label = f"Elasticity {df_updated['type'].iloc[0]}"
        df_updated.loc[condition, "details"] = (
            df_updated.loc[condition, "details"].astype(str) + f" | {elasticity_label}"
        )

    return df_updated


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


def handle_empty_group(df_results: pd.DataFrame) -> pd.DataFrame:
    """Handles the case where the group DataFrame is empty.

    Args:
        df_results (pd.DataFrame): The DataFrame containing the overall results.

    Returns:
        pd.DataFrame: Updated `df_results` with a default type and result to push.
    """
    logging.error("No group elasticity data available.")
    df_results["result_to_push"] = df_results["quality_test"]
    df_results["type"] = "uid"
    return df_results


def process_group_results(
    df_group: pd.DataFrame, segmentation_column: str, data_columns: DataColumns, group_type: str
) -> Optional[pd.DataFrame]:
    """Processes segmentation group results for elasticity.

    Args:
        df_group (pd.DataFrame): The DataFrame containing group data.
        segmentation_column (str): The column used for segmentation.
        data_columns (DataColumns): Configuration of data columns.
        group_type (str): The type label for this group (e.g., 'group 1', 'group 2').

    Returns:
        pd.DataFrame: The processed group DataFrame with the specified type.
    """
    if segmentation_column in df_group.columns:
        df_results_group = process_group_segmentation(
            df_group=df_group, segmentation_column=segmentation_column, data_columns=data_columns
        )
        df_results_group["type"] = group_type
        return df_results_group

    logging.error(f"No {segmentation_column} column in in df_group.")
    return None


def set_flags_for_groups(
    df_results_group_1: pd.DataFrame,
    df_results: pd.DataFrame,
    df_results_group_2: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Sets the result_to_push flag  and details for group results.

    Args:
        df_results_group_1 (pd.DataFrame): Results for group 1.
        df_results (pd.DataFrame): Original results DataFrame.
        df_results_group_2 (Optional[pd.DataFrame]): Results for group 2. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
            - Updated DataFrame for group 1.
            - Updated DataFrame for group 2 if provided, else None.

    Raises:
        KeyError: If required columns are missing in any of the input DataFrames.
    """
    # Update group 1
    df_group_1_updated = set_result_flag(df_results_group_1, df_results)

    # Update group 2 if provided
    df_group_2_updated = None
    if df_results_group_2 is not None:
        df_group_2_updated = set_result_flag(df_results_group_2, df_results, df_group_1_updated)

    return df_group_1_updated, df_group_2_updated


def add_missing_uids(
    df_results: pd.DataFrame, df_by_price_all: pd.DataFrame, output_columns: List[str] = OUTPUT_CS
) -> pd.DataFrame:
    """Adds missing UIDs from df_by_price_all to df_results with default values.

    Args:
        df_results (pd.DataFrame): DataFrame containing existing results with a 'uid' column.
        df_by_price_all (pd.DataFrame): DataFrame containing all UIDs to compare against.
        output_columns (List[str]): List of columns in df_results to add, all set to NaN for
        new rows.

    Returns:
        pd.DataFrame: Updated df_results with new UIDs added and default values applied.
    """
    if "uid" not in df_results.columns:
        raise ValueError("'uid' column is missing in df_results.")
    if "uid" not in df_by_price_all.columns:
        raise ValueError("'uid' column is missing in df_by_price_all.")

    # Identify UIDs not already in df_results
    new_uids = df_by_price_all.loc[~df_by_price_all["uid"].isin(df_results["uid"]), "uid"]

    # Create a DataFrame with the new UIDs
    new_rows = pd.DataFrame({"uid": new_uids})

    # Add columns from output_columns and set values to np.nan, except for 'quality_test'
    for col in output_columns:
        new_rows[col] = np.nan

    # Set 'quality_test' to False for the new rows
    new_rows["quality_test"] = False
    new_rows["quality_test_high"] = False
    new_rows["quality_test_medium"] = False
    # Set s to False for the new rows
    new_rows["elasticity_level"] = ""
    new_rows["details"] = ""

    # Concatenate the new rows with the existing df_results
    return pd.concat([df_results, new_rows], ignore_index=True)


def add_group_elasticity(
    df_group: pd.DataFrame, df_results: pd.DataFrame, data_columns: DataColumns
) -> pd.DataFrame:
    """Run elasticity by group and consolidates results it into a single DataFrame.

    - Process group 1
    - Process group 2 if exist
    - Consolidate results:
        - updating type (uid, group 1, group 2) and details
        - result_to_push : True if one of the run passed the quality test.

    Args:
        df_group (pd.DataFrame): DataFrame containing segmentation data for user groups.
        df_results (pd.DataFrame): DataFrame with overall analysis results.
        data_columns (DataColumns): Configuration for column mappings.

    Returns:
        pd.DataFrame: Combined DataFrame with updated result_to_push flags,
        type and details.

    Raises:
        KeyError: If required columns are missing in the input DataFrames.
        ValueError: If processing fails due to unexpected data conditions.
    """
    if df_group.empty:
        return handle_empty_group(df_results)

    # Process group 1
    df_results_group_1 = process_group_results(
        df_group, "group_uid_segmentation_1", data_columns, group_type="group 1"
    )

    # Process group 2 if available
    df_results_group_2 = None
    if "group_uid_segmentation_2" in df_group.columns:
        df_results_group_2 = process_group_results(
            df_group, "group_uid_segmentation_2", data_columns, group_type="group 2"
        )

    # Process UID results
    df_results["type"] = "uid"
    df_results["result_to_push"] = df_results["quality_test"]

    # Set result_to_push flags
    df_results_group_1, df_results_group_2 = set_flags_for_groups(
        df_results_group_1, df_results, df_results_group_2
    )

    # Concatenate and return results
    all_results = [df_results, df_results_group_1]
    if df_results_group_2 is not None:
        all_results.append(df_results_group_2)

    return pd.concat(all_results, ignore_index=True)
