"""Module preprocessing for group elasticity."""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from elasticity.data.read import read_data_attrs
from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.s3 import io_tools as s3io

COMBINED_ATTR_COLUMN = "combined_attr_value"


def data_for_group_elasticity(
    df_by_price: pd.DataFrame, data_fetch_params: DataFetchParameters, end_date: str
) -> pd.DataFrame:
    """Generate group-level data for elasticity analysis.

    This function orchestrates the data fetching, merging, and transformation processes to prepare
    data for elasticity analysis.
    It creates group based on segmentation (if exist), attribute data (if exist)
    and price similarity.

    Args:
        df_by_price (pd.DataFrame): DataFrame containing data grouped by price.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        end_date (str): The end date for querying Athena.

    Returns:
        pd.DataFrame: DataFrame with group-level data for elasticity analysis.
                      Returns an empty DataFrame if attribute data fetching fails.
    """
    # Fetch segmentation data
    df_seg = fetch_segmentation_data(data_fetch_params.client_key, data_fetch_params.channel)

    # Merge segmentation data
    df_group, available_segments = merge_segmentation_data(df_by_price, df_seg)

    # Fetch and merge attribute data if attr_names are provided
    if data_fetch_params.attr_names:
        df_attrs = fetch_and_combine_attributes(data_fetch_params, end_date)
        if df_attrs.empty:
            logging.error("Attribute data fetching failed. Returning empty DataFrame.")
            return pd.DataFrame()  # Return empty DataFrame if attribute fetching fails
        df_group = merge_attribute_data(df_group, df_attrs)

    return process_groups(df_group, available_segments, data_fetch_params)


def process_groups(
    df_group: pd.DataFrame, available_segments: List[str], data_fetch_params: DataFetchParameters
) -> pd.DataFrame:
    """Process groups.

    Based on available segments and price group calculated by the function group_by_similarity.

    Args:
        df_group (pd.DataFrame): The DataFrame containing merged segmentation and price data.
        available_segments (List[str]): List of segmentation columns available for processing.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.

    Returns:
        pd.DataFrame: The processed DataFrame with unique group identifiers.
    """
    for segmentation_col in available_segments:
        sub_group_cols = [segmentation_col]
        group_cols = [segmentation_col, "price_group"]

        if data_fetch_params.attr_names:
            sub_group_cols.append(COMBINED_ATTR_COLUMN)
            group_cols.append(COMBINED_ATTR_COLUMN)

        df_group["sub_group_uid"] = df_group[sub_group_cols].astype(str).agg("_".join, axis=1)

        # Build group mapping
        group_mapping = {}
        unique_sub_groups = df_group["sub_group_uid"].unique()
        for gr in unique_sub_groups:
            subset = df_group[df_group["sub_group_uid"] == gr]
            group_mapping_dict = group_by_similarity(subset)
            group_mapping.update(group_mapping_dict)

        df_group["price_group"] = df_group["uid"].map(group_mapping)

        # Drop rows with missing group columns
        df_group = df_group.dropna(subset=group_cols)

        # Create a unique group identifier
        group_uid_col = f"group_uid_{segmentation_col}"
        df_group[group_uid_col] = df_group[group_cols].astype(str).agg("_".join, axis=1)

    return df_group


def merge_attribute_data(df_group: pd.DataFrame, df_attrs: pd.DataFrame) -> pd.DataFrame:
    """Merge attribute data into the main DataFrame.

    Args:
        df_group (pd.DataFrame): The main DataFrame to merge attributes into.
        df_attrs (pd.DataFrame): DataFrame containing 'uid' and
        combined attribute columns.

    Returns:
        pd.DataFrame: The merged DataFrame with attribute data.
    """
    if not df_attrs.empty:
        return df_group.merge(df_attrs, on="uid", how="left")
    return df_group


def merge_segmentation_data(
    df_by_price: pd.DataFrame, df_seg: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """Merge segmentation data with the main DataFrame.

    Args:
        df_by_price (pd.DataFrame): DataFrame containing data grouped by price.
        df_seg (pd.DataFrame): Segmentation DataFrame to merge with df_by_price.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Merged DataFrame (`df_group`).
            - List of available segmentation columns (`available_segments`).
    """
    if not df_seg.empty:
        df_group = df_by_price.merge(df_seg, on="uid", how="left")
        available_segments = ["segmentation_1", "segmentation_2"]
    else:
        df_group = df_by_price.copy()
        df_group["segmentation_1"] = "no_segmentation"
        available_segments = ["segmentation_1"]
    return df_group, available_segments


def fetch_and_combine_attributes(
    data_fetch_params: DataFetchParameters, end_date: str
) -> pd.DataFrame:
    """Fetch attribute data then combine attribute columns.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        end_date (str): The end date for querying Athena.

    Returns:
        pd.DataFrame: A DataFrame containing 'uid' and the combined attribute column.
                      Returns an empty DataFrame if fetching or combining fails.
    """
    try:
        date_params = {"end_date": end_date}
        df_attrs = read_data_attrs(data_fetch_params=data_fetch_params, date_params=date_params)
        # Combine columns from attr_names into a single column
        df_attrs[COMBINED_ATTR_COLUMN] = (
            df_attrs[data_fetch_params.attr_names].astype(str).agg("_".join, axis=1)
        )
        return df_attrs[["uid", COMBINED_ATTR_COLUMN]]
    except Exception as e:
        logging.error(f"Error getting attribute data: {e}")
        return pd.DataFrame()


def fetch_segmentation_data(client_key: str, channel: str) -> pd.DataFrame:
    """Fetch segmentation data based on client_key and channel.

    Args:
        client_key (str): The unique identifier for the client.
        channel (str): The communication channel identifier.

    Returns:
        pd.DataFrame: A DataFrame containing segmentation data.
        Returns an empty DataFrame if fetching fails.
    """
    try:
        return get_segmentation_data(client_key, channel)
    except Exception as e:
        logging.error(f"Error getting segmentation data: {e}")
        return pd.DataFrame()


def tag_segmentation(
    df_segmentation: pd.DataFrame, segment_col: str = "kvi_segment", uid_col: str = "uid"
) -> pd.DataFrame:
    """Tags rows of the dataframe based on segment values.

    Renames the specified segment column to 'segmentation_1' and creates a new column
    'segmentation_2' that categorizes rows based on the values in 'segmentation_1'
    by merging KVI and SD to KVI_SD.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the specified columns.
    segment_col (str): The name of the segment_col. Default is 'kvi_segment'.
    uid_col (str): The name of the uid column. Default is 'uid'.

    Returns:
    pd.DataFrame: The dataframe with columns 'uid', 'segmentation_1', and 'segmentation_2'.
    """
    df_segmentation = df_segmentation.rename(columns={segment_col: "segmentation_1"})

    df_segmentation["segmentation_2"] = np.where(
        np.isin(df_segmentation["segmentation_1"], ["KVI", "SD"]),
        "KVI_SD",
        df_segmentation["segmentation_1"],
    )

    return df_segmentation[[uid_col, "segmentation_1", "segmentation_2"]]


def get_segmentation_data(
    client_key: str,
    channel: str,
) -> pd.DataFrame:
    """Retrieves segmentation data from S3 based on the client key.

    Args:
        client_key (str): The client key.
        channel (str): The channel.

    Returns:
        pandas.DataFrame: The segmentation data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If no files are found for the given client key and channel.
    """
    s3_dir_article_segmentation = app_state.s3_eval_results_dir.replace(
        app_state.project_name, "article_segmentation"
    )
    df_seg = s3io.maybe_get_pd_csv_df(
        file_name=f"{client_key}_{channel}.csv",
        s3_dir=s3_dir_article_segmentation,
        usecols=["uid", "kvi_segment"],
    )

    if df_seg.empty:
        raise FileNotFoundError(
            f"No files found for client_key={client_key} and channel={channel} "
            f"in {s3_dir_article_segmentation}."
        )
    return tag_segmentation(df_seg)


def group_by_similarity(
    df: pd.DataFrame,
    median_percentage: float = 15,
    volatility_percentage: float = 30,
    uid_col: str = "uid",
    price_col: str = "round_price",
) -> dict[str, int]:
    """Groups data points in a DataFrame by UID based on similarity.

    Similarity in price median and volatility. Similarity is determined by thresholds
    defined as percentages of the current UID's median and volatility values.

    Args:
        df (pd.DataFrame): The DataFrame containing UID and price data.
        median_percentage (float, optional): The percentage threshold for median
            similarity. Defaults to 15.
        volatility_percentage (float, optional): The percentage threshold for
            volatility similarity. Defaults to 30.
        uid_col (str, optional): The column name containing unique identifiers
            (UIDs). Defaults to 'uid'.
        price_col (str, optional): The column name containing price data. Defaults
            to 'round_price'.

    Returns:
        dict[str, int]: A dictionary mapping UIDs to their assigned group ID.
    """
    # Calculate median and volatility for each UID
    stats_df = (
        df[~df["outlier_quantity"]]
        .groupby(uid_col)[price_col]
        .agg(median="median", volatility=lambda x: calculate_volatility(x))
        .reset_index()
    )

    # Sort statistics to enhance the grouping process
    stats_df = stats_df.sort_values(by=["median", "volatility"]).reset_index(drop=True)

    used = set()
    group_id = 0
    group_mapping = {}
    uids = stats_df[uid_col].to_numpy()
    medians = stats_df["median"].to_numpy()
    volatilities = stats_df["volatility"].to_numpy()

    for i in range(len(stats_df)):
        if uids[i] in used:
            continue

        # Calculate percentage thresholds for the current UID
        median_threshold = median_percentage / 100 * medians[i]
        volatility_threshold = volatility_percentage / 100 * volatilities[i]

        # Find UIDs within the threshold
        median_diff = abs(medians - medians[i])
        volatility_diff = abs(volatilities - volatilities[i])
        mask = (median_diff <= median_threshold) & (volatility_diff <= volatility_threshold)

        similar_uids = uids[mask & ~np.isin(uids, list(used))]

        # Update group_mapping and used set for similar UIDs
        group_mapping.update(dict.fromkeys(similar_uids, group_id))
        used.update(similar_uids)

        group_id += 1

    return group_mapping


def calculate_volatility(prices: list[float]) -> float:
    """Calculates the volatility of a list of prices.

    Calculates the volatility of a list of prices as the standard deviation
    divided by the mean price. Handles the case where the mean price is zero
    and returns 0 for volatility.

    Args:
        prices (list[float]): A list of prices.

    Returns:
        float: The volatility of the prices.
    """
    mean_price = prices.mean()
    std_dev = prices.std()
    if mean_price == 0:
        return 0
    return std_dev / mean_price
