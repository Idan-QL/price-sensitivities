"""Module preprocessing for group elasticity."""

import logging
import re
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from elasticity.data.read import read_data_attrs
from ql_toolkit.s3 import io as s3io
from ql_toolkit.s3 import ls as s3ls


def generate_segmentation_dfs(df_group: pd.DataFrame, df_results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate DataFrames containing counts of `elasticity_level` values grouped by segmentation columns.

    This function merges two DataFrames (`df_group` and `df_results`), then counts the occurrences of
    `elasticity_level` values for each unique value in `group_uid_segmentation_1` and `group_uid_segmentation_2`.
    The counts are aggregated into two separate DataFrames, one for each segmentation column.

    Parameters:
    - df_group (pd.DataFrame): DataFrame with columns 'uid', 'group_uid_segmentation_1', and 'group_uid_segmentation_2'.
    - df_results (pd.DataFrame): DataFrame with columns 'uid', 'elasticity_level', and 'quality_test'.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
      - df_segmentation_1: DataFrame with counts of `elasticity_level` values grouped by 'group_uid_segmentation_1'.
      - df_segmentation_2: DataFrame with counts of `elasticity_level` values grouped by 'group_uid_segmentation_2'.
    """
    
    # Merge df_group and df_results on 'uid'
    df_group_homogenity = df_group[['uid', 'group_uid_segmentation_1', 'group_uid_segmentation_2']].drop_duplicates() \
        .merge(df_results[['uid', 'elasticity_level', 'quality_test']], on='uid')

    # Function to count occurrences of each elasticity_level, including NaNs
    def count_elasticity(df: pd.DataFrame) -> Dict[str, int]:
        """
        Count occurrences of each `elasticity_level`, including NaN values.

        Parameters:
        - df (pd.DataFrame): DataFrame with 'elasticity_level' column.

        Returns:
        - Dict[str, int]: Dictionary where keys are `elasticity_level` values (including 'NaN') and values are counts.
        """
        counts = df['elasticity_level'].fillna('NaN').value_counts()
        return counts.to_dict()

    # Generate df_segmentation_1
    df_segmentation_1 = df_group_homogenity.groupby('group_uid_segmentation_1').apply(
        lambda x: pd.Series({
            'uid_elasticity': count_elasticity(x)
        })
    ).reset_index()

    # Generate df_segmentation_2
    df_segmentation_2 = df_group_homogenity.groupby('group_uid_segmentation_2').apply(
        lambda x: pd.Series({
            'uid_elasticity': count_elasticity(x)
        })
    ).reset_index()

    return df_segmentation_1, df_segmentation_2


def data_for_group_elasticity(
    df_by_price: pd.DataFrame,
    client_key: str,
    channel: str,
    end_date: str,
    attr_name: Optional[str] = None,
) -> pd.DataFrame:
    """Generate group-level data for elasticity analysis.

    Args:
        df_by_price (pandas.DataFrame): DataFrame containing data grouped by price.
        client_key (str): Client key for segmentation data.
        channel (str): Channel for segmentation data.
        end_date (str): end_date for querying athena.
        attr_name (str, optional): Column name for attribute data. Defaults None.

    Returns:
        pandas.DataFrame: DataFrame with group-level data for elasticity analysis.
                          If an error occurs in retrieving segmentation or attribute data,
                          an empty DataFrame is returned.
    """
    try:
        df_seg = get_segmentation_data(client_key, channel)
    except Exception as e:
        logging.error(f"Error getting segmentation data: {e}")
        return pd.DataFrame()

    df_group = df_by_price.merge(df_seg, on="uid", how="left")

    if attr_name:
        try:
            attr_col = "attr_value"
            date_params = {"end_date": end_date}
            df_attrs = read_data_attrs(
                client_key, channel, attr_names=[attr_name], date_params=date_params
            )
            df_group = df_group.merge(df_attrs, on="uid", how="left")
        except Exception as e:
            logging.error(f"Error getting attr. data: {e}")
            return pd.DataFrame()

    for segmentation_col in ["segmentation_1", "segmentation_2"]:
        sub_group_cols = [segmentation_col]
        group_cols = [segmentation_col, "price_group"]
        if attr_name:
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
    df: pd.DataFrame,
    revenue_col: str = "total_revenue",
    rank_col: str = "ranking_score_rank",
    uid_col: str = "uid",
    threshold_KVI: float = 0.25,
    threshold_SD: float = 0.75,
) -> pd.DataFrame:
    """Tags rows of the dataframe based on accumulated revenue.

    Parameters:
    df (pd.DataFrame): The input dataframe with columns 'uid', 'ranking_score',
    'ranking_score_rank', and 'total_revenue'.
    revenue_col (str): The name of the revenue column. Default is 'total_revenue'.
    rank_col (str): The name of the rank column. Default is 'ranking_score_rank'.
    threshold_KVI (float): The first threshold as a fraction of total revenue.
    Default is 0.25 (25%).
    threshold_SD (float): The second threshold as a fraction of total revenue.
    Default is 0.75 (75%).

    Returns:
    pd.DataFrame: The dataframe with an additional 'tag' column.
    """
    # Step 1: Sort the dataframe by `ranking_score_rank`
    df_sorted = df.sort_values(by=rank_col)

    # Step 2: Calculate cumulative revenue and determine the revenue thresholds
    df_sorted["cumulative_revenue"] = df_sorted[revenue_col].cumsum()
    total_revenue = df_sorted[revenue_col].sum()
    threshold_revenue_KVI = total_revenue * threshold_KVI
    threshold_revenue_SD = total_revenue * threshold_SD

    # Step 3: Tag rows based on cumulative revenue
    df_sorted["segmentation_1"] = pd.cut(
        df_sorted["cumulative_revenue"],
        bins=[-float("inf"), threshold_revenue_KVI, threshold_revenue_SD, float("inf")],
        labels=["KVI", "SD", "PG"],
    )
    df_sorted["segmentation_2"] = pd.cut(
        df_sorted["cumulative_revenue"],
        bins=[-float("inf"), threshold_revenue_SD, float("inf")],
        labels=["KVI_SD", "PG"],
    )

    return df_sorted[[uid_col, "segmentation_1", "segmentation_2"]]


def get_latest_file(
    file_list: list[str],
) -> str:
    """Gets the file path from the list with the most recent date based on the filename.

    Args:
        file_list (list[str]): A list of file paths.

    Returns:
        str: The file path with the most recent date.
    """
    latest_file = None
    latest_date = None

    for file in file_list:
        # Extract the date portion from the filename
        date_str = file.split("/")[-1].split("_")[-1].replace(".csv", "")

        # Convert the date string to a datetime object
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            # Handle invalid date format cases (optional)
            continue

        # Update latest_file and latest_date if the current date is newer
        if not latest_date or date > latest_date:
            latest_file = file
            latest_date = date

    return latest_file


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
    s3_files = s3ls.list_files_in_dir(s3_dir=s3_dir, file_type="csv")
    filtered_files_list = [
        item
        for item in s3_files
        if f"{client_key}_{channel}" in item and "_consts" not in item and "backups" not in item
    ]

    if not filtered_files_list:
        raise FileNotFoundError(
            f"No files found for client_key={client_key} and channel={channel} in {s3_dir}."
        )

    file_name = get_latest_file(filtered_files_list)
    df_seg = s3io.maybe_get_pd_csv_df(
        file_name=file_name,
        s3_dir="",
        usecols=["uid", "ranking_score", "ranking_score_rank", "total_revenue"],
    )
    return tag_segmentation(df_seg)


def validate_single_word(param: str) -> None:
    """Validate that the parameter is a valid single word."""
    if not re.match(r"^\w+$", param):
        raise ValueError(
            f"Pyathena query: '{param}' must be a single word containing "
            "only alphanumeric characters and underscores."
        )


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
