"""Module of writing graphs."""

from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
from ql_toolkit.s3 import io as s3io


def save_distribution_graph(client_key: str,
                            channel: str,
                            total_uid: int,
                            df_report: pd.DataFrame,
                            end_date:str, s3_dir: str) -> None:
    """Save a graph representing the distribution of UID with elasticity.

    Parameters:
        client_key (str): The client key.
        channel (str): The channel.
        total_end_date_uid (int): The total number of UID.
        df_results (pd.DataFrame): DataFrame containing the results.
        s3_dir (str): The S3 directory to upload the graph to.

    Returns:
        None
    """
    # Calculate the percentage of UID with elasticity out of the total UID
    total_uid_with_elasticity = df_report['uid_with_elasticity'].sum()
    total_uid_with_data_for_elasticity =  df_report['uid_with_data_for_elasticity'].sum()
    total_percentage_all = total_uid_with_elasticity / total_uid * 100
    total_percentage_with_data = (total_uid_with_elasticity
                                  / total_uid_with_data_for_elasticity * 100)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Data
    columns = ["uid_with_elasticity_less_than_minus6",
               "uid_with_elasticity_moreorequal_minus6_less_than_minus3.8",
               'uid_with_elasticity_moreorequal_minus3.8_less_than_minus1',
               'uid_with_elasticity_moreorequal_minus1_less_than_0',
               'uid_with_elasticity_moreorequal_0_less_than_1',
               'uid_with_elasticity_moreorequal_1_less_than_3.8',
               "uid_with_elasticity_moreorequal_3.8_less_than_6",
               "uid_with_elasticity_moreorequal_6"]
    labels = ["<-6", "[-6;-3.8[", "[-3.8;-1[", "[-1;0[", "[0;1[", "[1;3.8[", "[3.8;6[", "6+"]
    counts = [df_report[column].sum() for column in columns]

    # Plot
    bars = ax.bar(labels, counts)

    # Add numbers on top of bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, count, ha='center', va='bottom')

    # Add title and labels
    title = (f"{client_key}-{channel}-{total_uid} skus\n"
            f"{total_uid_with_elasticity} skus with elasticity "
            f"({total_percentage_all:.1f}% from total skus, "
            f"{total_percentage_with_data:.1f}% from skus with data)")
    ax.set_title(title)
    ax.set_xlabel('Elasticity')
    ax.set_ylabel('UID Count')

    # Save the plot to bytes buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Upload the plot to S3
    file_name = f"distribution_{client_key}_{channel}_{end_date}.png"


    s3io.upload_to_s3(s3_dir=s3_dir,
                      file_name=file_name,
                      file_obj=buffer)

    plt.close(fig)  # Close the figure to free memory
