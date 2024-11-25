"""Module of writing graphs."""

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from ql_toolkit.aws_data_management.s3 import io_tools as s3io

plt.switch_backend("agg")


def save_distribution_graph(
    data_fetch_params: DataFetchParameters,
    total_uid: int,
    df_report: pd.DataFrame,
    end_date: str,
    s3_dir: str,
) -> None:
    """Save a graph representing the distribution of UID with elasticity.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        total_uid (int): The total number of UID.
        df_report (pd.DataFrame): DataFrame containing the report results.
        end_date (str): The end date.
        s3_dir (str): The S3 directory to upload the graph to.

    Returns:
        None
    """
    # Calculate the percentage of UID with elasticity out of the total UID
    total_uid_with_elasticity = df_report["uids_with_elasticity"].sum()
    uids_for_elasticity_calculation = df_report["uids_with_elasticity_data"].sum()
    total_percentage_all = total_uid_with_elasticity / total_uid * 100
    total_percentage_with_data = total_uid_with_elasticity / uids_for_elasticity_calculation * 100

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Data: updated column names based on report_row structure
    columns = [
        "uids_below_minus6",
        "uids_between_minus6_and_min",
        "uids_between_min_and_minus1",
        "uids_between_minus1_and_0",
        "uids_equal_0",
        "uids_above_0",
    ]
    labels = [
        "<-6",
        "[-6;min[",
        "[min;-1[",
        "[-1;0[",
        "0",
        ">0",
    ]
    counts = [df_report[column].sum() for column in columns]

    # Plot
    bars = ax.bar(labels, counts)

    # Add numbers on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, count, ha="center", va="bottom")

    # Add title and labels
    title = (
        f"{data_fetch_params.client_key}-{data_fetch_params.channel}-{total_uid} skus\n"
        f"{total_uid_with_elasticity} skus with elasticity "
        f"({total_percentage_all:.1f}% from total skus, "
        f"{total_percentage_with_data:.1f}% from skus with data)"
    )
    ax.set_title(title)
    ax.set_xlabel("Elasticity")
    ax.set_ylabel("UID Count")

    # Save the plot to bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Upload the plot to S3
    file_name = (
        f"distribution_{data_fetch_params.client_key}_{data_fetch_params.channel}_{end_date}.png"
    )

    s3io.upload_to_s3(s3_dir=s3_dir, file_name=file_name, file_obj=buffer)

    plt.close(fig)  # Close the figure to free memory
