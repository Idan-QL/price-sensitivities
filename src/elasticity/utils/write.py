"""Module for writing results and monitoring outputs to S3 and Athena."""

import logging
from datetime import datetime

import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.data_lake.athena_glue_operations import AthenaDataManager
from ql_toolkit.data_lake.data_classes import GlueDBKeys


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
    df[numeric_cols] = df[numeric_cols].replace([pd.NA, float("inf"), -float("inf")], 0)

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
    data_fetch_params: DataFetchParameters,
    end_date: str,
    df_upload: pd.DataFrame,
    table_name: str,
    is_qa_run: bool,
) -> None:
    """Upload elasticity data to an Athena partition using AthenaDataManager.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        end_date (str): The date to be used for partitioning.
        df_upload (pd.DataFrame): The DataFrame containing elasticity data.
        table_name (str): The name of the Glue table.
        is_qa_run (bool): Save is_qa_run value.

    Returns:
        None
    """
    # Initialize GlueDBKeys dataclass with relevant values
    glue_db_keys = GlueDBKeys(
        database_name=app_state.athena_database_name,
        table_name=table_name,
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        date=datetime.today().date().strftime("%Y-%m-%d"),
    )

    # Define columns to add if missing
    columns_to_add = {
        "data_date": end_date,
        "date": datetime.today().date(),
        "project_name": app_state.project_name,
        "client_key": data_fetch_params.client_key,
        "channel": data_fetch_params.channel,
    }

    # Add run_type
    columns_to_add["run_type"] = "QA" if is_qa_run else "Production"

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
