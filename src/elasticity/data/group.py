"""Module preprocessing for group elasticity."""

import logging

import numpy as np
import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from elasticity.data.read import read_data_attrs
from ql_toolkit.s3 import io_tools as s3io


def data_for_group_elasticity(
    df_by_price: pd.DataFrame, data_fetch_params: DataFetchParameters, end_date: str
) -> pd.DataFrame:
    """Generate group-level data for elasticity analysis.

    Args:
        df_by_price (pandas.DataFrame): DataFrame containing data grouped by price.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        end_date (str): end_date for querying athena.

    Returns:
        pandas.DataFrame: DataFrame with group-level data for elasticity analysis.
                          If an error occurs in retrieving segmentation or attribute
                          data, an empty DataFrame is returned.
    """
    try:
        df_seg = get_segmentation_data(data_fetch_params.client_key, data_fetch_params.channel)
    except Exception as e:
        logging.error(f"Error getting segmentation data: {e}")
        return pd.DataFrame()

    df_group = df_by_price.merge(df_seg, on="uid", how="left")

    if data_fetch_params.attr_name:
        try:
            attr_col = "attr_value"
            date_params = {"end_date": end_date}
            df_attrs = read_data_attrs(data_fetch_params=data_fetch_params, date_params=date_params)
            df_group = df_group.merge(df_attrs, on="uid", how="left")
        except Exception as e:
            logging.error(f"Error getting attr. data: {e}")
            return pd.DataFrame()

    for segmentation_col in ["segmentation_1", "segmentation_2"]:
        sub_group_cols = [segmentation_col]
        group_cols = [segmentation_col, "price_group"]
        if data_fetch_params.attr_name:
            sub_group_cols.append(attr_col)
            group_cols.append(attr_col)

        df_group["sub_group_uid"] = df_group[sub_group_cols].apply(
            lambda row: "_".join(row.to_numpy().astype(str)), axis=1
        )

        group_mapping = {}
        for gr in df_group["sub_group_uid"].unique():
            group_mapping_gr = group_by_similarity(df_group[df_group["sub_group_uid"] == gr])
            group_mapping = {**group_mapping, **group_mapping_gr}

        df_group["price_group"] = df_group["uid"].map(group_mapping)

        df_group = df_group.dropna(subset=group_cols)
        df_group["group_uid_" + segmentation_col] = df_group[group_cols].apply(
            lambda row: "_".join(row.to_numpy().astype(str)), axis=1
        )

    return df_group


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
    s3_dir: str = "data_science/eval_results/article_segmentation/",
) -> pd.DataFrame:
    """Retrieves segmentation data from S3 based on the client key.

    Args:
        client_key (str): The client key.
        channel (str): The channel.
        s3_dir (str, optional): The S3 directory where the data is stored.
        Defaults to 'data_science/eval_results/article_segmentation/'.

    Returns:
        pandas.DataFrame: The segmentation data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If no files are found for the given client key and channel.
    """
    df_seg = s3io.maybe_get_pd_csv_df(
        file_name=f"{client_key}_{channel}.csv",
        s3_dir=s3_dir,
        usecols=["uid", "kvi_segment"],
    )

    if df_seg.empty:
        raise FileNotFoundError(
            f"No files found for client_key={client_key} and channel={channel} in {s3_dir}."
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
