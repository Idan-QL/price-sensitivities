"""Module for writing results and monitoring outputs to S3 and Athena."""

import logging
from datetime import datetime

import pandas as pd

from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.data_lake.athena_glue_operations import AthenaDataManager
from ql_toolkit.data_lake.data_classes import GlueDBKeys
from ql_toolkit.s3.io_tools import write_dataframe_to_s3


def add_missing_columns(df: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Add missing columns to the DataFrame with the specified values.

    Args:
        df (pd.DataFrame): The DataFrame to check and add columns to.
        columns (dict): A dictionary where keys are column names and values are the default values
        to assign.

    Returns:
        pd.DataFrame: The DataFrame with missing columns added.
    """
    for column, value in columns.items():
        if column not in df.columns:
            df[column] = value
    return df


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric_columns.

    Clean all infinite or NaN values in the numeric columns of the DataFrame and attempt to convert
    columns that can be represented as integers. Logs the columns that are successfully converted.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.

    Returns:
        pd.DataFrame: A DataFrame with NaN, infinite, and -infinite values replaced by 0, and
                      numeric columns converted to integer type where possible.
    """
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    df[numeric_cols] = df[numeric_cols].replace([pd.NA, pd.NaT, float("inf"), -float("inf")], 0)

    converted_cols = []  # List to store columns that are converted to int

    # Attempt to convert columns to integer where possible
    for col in numeric_cols:
        if pd.api.types.is_float_dtype(df[col]):
            # Replace any remaining NaN values with 0 to ensure safe conversion
            df[col] = df[col].fillna(0)

            # Check if it is safe to convert to int (i.e., no data loss in conversion)
            try:
                if (df[col] == df[col].astype(int)).all():
                    df[col] = df[col].astype(int)
                    converted_cols.append(col)  # Add column to the list of converted columns
            except pd.errors.IntCastingNaNError:
                logging.warning(f"Column {col} contains NaN or inf, skipping conversion to int.")

    # Log the columns that were converted to int
    if converted_cols:
        logging.info(f"Columns successfully converted to int: {', '.join(converted_cols)}")
    else:
        logging.info("No columns were converted to int.")

    return df


def upload_elasticity_data_to_athena(
    client_key: str, channel: str, end_date: str, df_upload: pd.DataFrame, table_name: str
) -> None:
    """Upload elasticity data to an Athena partition using AthenaDataManager.

    Args:
        client_key (str): The client key to be used for partitioning.
        channel (str): The channel to be used for partitioning.
        end_date (str): The date to be used for partitioning.
        df_upload (pd.DataFrame): The DataFrame containing elasticity data.
        table_name (str): The name of the Glue table.

    Returns:
        None
    """
    # Initialize GlueDBKeys dataclass with relevant values
    glue_db_keys = GlueDBKeys(
        database_name=app_state.athena_database_name,
        table_name=table_name,
        client_key=client_key,
        channel=channel,
        date=datetime.today().date().strftime("%Y-%m-%d"),
    )

    # Define columns to add if missing
    columns_to_add = {
        "data_date": end_date,
        "date": datetime.today().date(),
        "project_name": app_state.project_name,
        "client_key": client_key,
        "channel": channel,
    }

    # Add missing columns to df_upload
    df_upload = add_missing_columns(df=df_upload, columns=columns_to_add)

    # Clean numeric columns
    df_upload = clean_numeric_columns(df=df_upload)

    # Initialize AthenaDataManager
    athena_data_manager = AthenaDataManager(
        region_name=app_state.s3_region,
        database_s3_uri=app_state.athena_database_s3_uri,
        glue_db_keys_dc=glue_db_keys,
    )

    # Upload DataFrame to Athena partition
    athena_data_manager.upload_dataframe_to_partition(data_df=df_upload)

    # Write CSV file - interim solution for CSMs
    write_dataframe_to_s3(
        file_name=f"{client_key}_{channel}_{table_name}_{end_date}.csv",
        xdf=df_upload,
        s3_dir=app_state.s3_eval_results_dir,
        rename_old=False,
    )
