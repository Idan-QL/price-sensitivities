"""Module for writing results and monitoring outputs to S3 and Athena."""

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.data_catalog.athena_glue_db_operations import (
    AthenaDataManager,
)
from ql_toolkit.aws_data_management.data_catalog.pydantic_models import (
    AthenaUploadConfig,
    GlueDBKeys,
)


def upload_data_to_athena(config: AthenaUploadConfig, df_upload: pd.DataFrame) -> None:
    """Upload data to an Athena partition using AthenaDataManager.

    This function prepares and uploads a pandas DataFrame to an Athena table by:
    - Adding missing columns required for the Athena table, with default values.
    - Uploading the processed DataFrame to the appropriate Athena partition using
      the `AthenaDataManager`.

    Args:
        config (AthenaUploadConfig): Configuration object containing:
            - client_key (str): Identifier for the client.
            - channel (str): Channel associated with the data.
            - table_name (str): Name of the Athena table to upload the data to.
        df_upload (pd.DataFrame): The pandas DataFrame to upload, containing the data
                                  to be processed and stored in Athena.

    Returns:
        None
    """
    columns_to_add = {
        "date": datetime.now(tz=UTC).date(),
        "client_key": config.client_key,
        "channel": config.channel,
        "project_name": app_state.project_name,
    }

    df_upload = add_missing_columns(df=df_upload, columns=columns_to_add)

    # TODO: define types per columns
    # df_upload = convert_columns(
    #     df=df_upload,
    #     nan_fill_value=config.nan_fill_value,
    #     inf_fill_value=config.inf_fill_value,
    # )

    glue_db_keys = GlueDBKeys(
        database_name=app_state.athena_database_name,
        table_name=config.table_name,
        client_key=config.client_key,
        channel=config.channel,
        date=datetime.now(tz=UTC).strftime("%Y-%m-%d"),
    )

    athena_data_manager = AthenaDataManager(
        region_name=app_state.aws_region,
        database_s3_uri=app_state.athena_database_s3_uri,
        glue_db_keys_dc=glue_db_keys,
    )

    athena_data_manager.upload_dataframe_to_partition(data_df=df_upload)


def add_missing_columns(df: pd.DataFrame, columns: dict[str, Any]) -> pd.DataFrame:
    """Add missing columns to the DataFrame with the specified values.

    Args:
        df (pd.DataFrame): The DataFrame to check and add columns to.
        columns (dict[str, Any]): A dictionary where keys are column names and
        values are the default values to assign.

    Returns:
        pd.DataFrame: The DataFrame with missing columns added.
    """
    num_rows = len(df)
    for column, value in columns.items():
        if column not in df.columns:
            df[column] = [value] * num_rows

    return df


def convert_numeric_column(col: pd.Series, nan_fill_value: int, inf_fill_value: int) -> pd.Series:
    """Convert numeric column, handling NaN and inf values explicitly.

    This function ensures that:
    - NaN values are replaced with a specified nan_fill_value.
    - Infinite values (positive and negative) are replaced with a specified inf_fill_value.
    - If all values in the column are integer-like after handling NaN and inf,
      the column is converted to int.
    - Otherwise, the column is converted to `float32`.

    Args:
        col (pd.Series): The input numeric column to process.
        nan_fill_value (int): The value to replace NaN entries.
        inf_fill_value (int): The value to replace inf/-inf entries.

    Returns:
        pd.Series: Processed column.
    """
    if col.isna().any():
        logging.warning(f"Column {col.name} contains NaN, replacing with {nan_fill_value}.")
        col = col.fillna(nan_fill_value)

    if np.isinf(col).any():
        logging.warning(f"Column {col.name} contains inf, replacing with {inf_fill_value}.")
        col = col.replace([np.inf, -np.inf], inf_fill_value)

    if (col == col.astype(int)).all():
        logging.info(f"Float column {col.name} successfully converted to int.")
        return col.astype(int)

    return col.astype("float32")


def convert_date_column(col: pd.Series) -> pd.Series:
    """Process a date column: convert it to string using app_state.date_format.

    Args:
        col (pd.Series): The input date column to process.

    Returns:
        pd.Series: Processed column as strings.
    """
    logging.info(f"Date column {col} successfully converted to string.")
    return col.dt.strftime(app_state.date_format)


def convert_columns(df: pd.DataFrame, nan_fill_value: int, inf_fill_value: int) -> pd.DataFrame:
    """Convert numeric and date columns in a DataFrame.

    This function performs the following transformations:
    - Handles NaN values in numeric columns by replacing them with a specified nan_fill_value.
    - Handles infinite values (positive and negative) in numeric columns by replacing them with a
      specified inf_fill_value.
    - Converts numeric float columns:
      - To integers if all values in the column are integer-like after processing.
      - To float32 if the column contains non-integer-like values.
    - Converts date columns to string format using app_state.date_format.

    Args:
        df (pd.DataFrame): The input DataFrame to process.
        nan_fill_value (int): The value to replace NaN entries in numeric columns.
        inf_fill_value (int): The value to replace inf/-inf entries in numeric columns.
        date_format (str): The format to use for date column conversion.

    Returns:
        pd.DataFrame: Processed DataFrame with modified numeric and date columns.
    """
    # Convert numeric columns
    float_cols = df.select_dtypes(include=["float"]).columns
    for col in float_cols:
        df[col] = convert_numeric_column(
            col=df[col], nan_fill_value=nan_fill_value, inf_fill_value=inf_fill_value
        )

    # Convert date columns
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
    for col in date_cols:
        df[col] = convert_date_column(col=df[col])

    return df
