"""This module contains functions for reading and writing Arrow tables using Polars DataFrames."""

import logging
from io import BytesIO
from typing import Optional

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from pyarrow import Table

from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.s3 import io_tools as s3io


def get_arrow_table(
    path_or_paths: str | list[str],
    s3_fs: s3fs.S3FileSystem = None,
    columns: Optional[list[str]] = None,
    filters: Optional[list[tuple]] = None,
) -> Table:
    """Retrieves an Arrow table from a Parquet file located at the specified path or paths.

    See arrow.apache.org/docs/python/generated/
        pyarrow.parquet.ParquetDataset.html#pyarrow.parquet.ParquetDataset
    for more details.

    Args:
        path_or_paths (str or list[str]): The path or list of paths to the Parquet file(s) to read.
        s3_fs (s3fs.S3FileSystem, default None): The S3FileSystem to read the Parquet file(s).
        columns (list[str], default None): The list of columns to read from the Parquet file(s).
        filters (list[tuple], default None): The list of filters to apply to the Parquet file(s).

    Returns:
        pyarrow.Table: The Arrow table representing the data in the Parquet file(s).

    Raises:
        ValueError: If the provided path is invalid or the Parquet file(s) cannot be read.

    Example:
        table = get_arrow_table("data_science/archive/dataset.parquet")
    """
    table = None
    if s3_fs is None:
        s3_fs = s3fs.S3FileSystem()

    path_or_paths = add_bucket_name_to_paths(path_or_paths)

    try:
        pq_ds = pq.ParquetDataset(path_or_paths=path_or_paths, filesystem=s3_fs, filters=filters)
        if columns:
            pq_dataset_columns = pq_ds.schema.names
            logging.info(f"Parquet dataset available columns: {pq_dataset_columns}")
            columns = list(set(columns).intersection(set(pq_dataset_columns)))
            logging.info(f"Reading columns: {columns}")
        table = pq_ds.read(columns=columns)
    except ValueError as err:
        logging.error(f"ValueError caught: {err} for the paths list")
    except Exception as err:
        logging.error("Err caught: %s", err)
    return table


def add_bucket_name_to_paths(path_or_paths: str | list[str]) -> list[str]:
    """Returns the path_or_paths as a lit, in case it is a string.

    The app_state.bucket_name is prepended to each path in the list.

    Args:
        path_or_paths (str or list[str]): The path or list of paths to the Parquet file(s) to read.

    Returns:
        list[str]: The list of paths with the bucket name prepended.
    """
    if not isinstance(path_or_paths, list):
        path_or_paths = [path_or_paths]

    return [f"{app_state.bucket_name}/{path}" for path in path_or_paths]


def get_xf_from_arrow_table(
    path_or_paths: str | list[str],
    is_lazy: bool,
    s3_fs: s3fs.S3FileSystem = None,
    columns: Optional[list[str]] = None,
    filters: Optional[list[tuple]] = None,
) -> Optional[pl.DataFrame | pl.LazyFrame]:
    """Get a Polars DataFrame or LazyFrame from an Arrow table obtained from Parquet file(s).

    Retrieves a Polars DataFrame or LazyFrame from an Arrow table obtained from
    Parquet file(s) located at the specified path or paths.

    For more details, see:
    https://arrow.apache.org/docs/python/generated/
    pyarrow.parquet.ParquetDataset.html#pyarrow.parquet.ParquetDataset

    Args:
        path_or_paths (str or list[str]): The path or list of paths to the Parquet file(s) to read.
        is_lazy (bool): Specifies whether to return a LazyFrame (True) or DataFrame (False).
        s3_fs (s3fs.S3FileSystem, default None): The S3FileSystem to use to read Parquet file(s).
        columns (list[str], default None): The list of columns to read from the Parquet file(s).
        filters (list[tuple], default None): The list of filters to apply to the Parquet file(s).

    Returns:
        Optional[pl.DataFrame or pl.LazyFrame]: The Polars DataFrame or LazyFrame representing
        the data in the Parquet file(s), or None if there was an error during the conversion.

    Raises:
        ValueError: If the provided path(s) is invalid or the Parquet file(s) cannot be read.

    Example:
        dataframe = get_xf_from_arrow_table("data_science/archive/dataset.parquet", False)
    """
    table = get_arrow_table(
        path_or_paths=path_or_paths, s3_fs=s3_fs, columns=columns, filters=filters
    )
    if table is None:
        return None
    try:
        if is_lazy:
            return pl.from_arrow(table).lazy()
        return pl.from_arrow(table)
    except ValueError as err:
        logging.error("ValueError caught: %s for %s", err, path_or_paths.split("/")[-1])

    except Exception as err:
        logging.error("Err caught: %s", err)
    return None


def load_xf_from_list_of_parquet_files(
    s3_files_list: list[str],
    is_lazy: bool,
    columns: Optional[list[str]] = None,
    filters: Optional[list[tuple]] = None,
) -> Optional[pl.DataFrame | pl.LazyFrame]:
    """Loads a Polars DataFrame or LazyFrame from a list of Parquet files located at the `paths`.

    This function reads each Parquet file in the provided list using the
    `get_xf_from_arrow_table` function,
    and appends the resulting DataFrame or LazyFrame to a list. If the list of DataFrames or
    LazyFrames is not empty,
    the function concatenates them into a single DataFrame or LazyFrame and returns it.
    Otherwise, it returns None.

    Args:
        s3_files_list (list[str]): The list of paths to the Parquet files to read.
        is_lazy (bool): Specifies whether to return a LazyFrame (True) or DataFrame (False).
        columns (list[str], default None): The list of columns to read from the Parquet files.
        filters (list[tuple], default None): The list of filters to apply to the Parquet files.

    Returns:
        Optional[pl.DataFrame or pl.LazyFrame]: The concatenated Polars DataFrame or LazyFrame
        representing the data in the Parquet files, or None if no Parquet files were loaded.

    Example:
        dataframe = load_xf_from_list_of_parquet_files(
            s3_files_list = ["data_science/archive/dataset1.parquet",
            "data_science/archive/dataset2.parquet"],
            is_lazy = False
        )
    """
    s3_fs = s3fs.S3FileSystem()
    logging.info("Loading %s Parquet files from S3...", len(s3_files_list))
    xf_list = []
    for i, pq_filepath in enumerate(s3_files_list):
        # log info message every 5 files
        if i % 5 == 0:
            logging.info("Reading Parquet file %s of %s", i + 1, len(s3_files_list))
        xf = get_xf_from_arrow_table(
            path_or_paths=pq_filepath,
            is_lazy=is_lazy,
            s3_fs=s3_fs,
            columns=columns,
            filters=filters,
        )
        xf_list.append(xf)
    if xf_list:
        logging.info("Loaded %s Parquet files from S3. Concatenating...", len(xf_list))
        return pl.concat(xf_list)

    logging.error("No Parquet files were loaded from S3.")
    return None


def write_xf_to_s3(
    xf: pl.DataFrame | pl.DataFrame,
    path: str,
    backup_path: Optional[str] = None,
    rename_old: bool = True,
) -> None:
    """Writes a Polars DataFrame or LazyFrame to a Parquet file located at the specified path.

    If the file already exists, it will be renamed and moved to the backup path.

    Args:
        xf (pl.DataFrame or pl.LazyFrame): The Polars DataFrame or LazyFrame to write.
        path (str): The path to the Parquet file to write.
        backup_path (str): The path to the backup Parquet file to write.
        rename_old (bool, default True): Specifies whether to rename the old file (as backup) or
        overwrite it.

    Raises:
        ValueError: If the provided path is invalid or the DataFrame or LazyFrame cannot be written.

    Example:
        dataframe = pl.DataFrame({
            "name": ["Bob", "Alice"],
            "age": [25, 26]
        })
        write_xf_to_s3(dataframe, "data_science/archive/dataset.parquet")
    """
    if isinstance(xf, pl.LazyFrame):
        xf = xf.collect()
    if backup_path is None:
        # insert the string "backup" before the file extension
        backup_path = path[:-8] + "_backup" + path[-8:]
    s3io.write_dataframe_to_s3(
        file_name=path, xdf=xf, rename_old=rename_old, backup_path=backup_path
    )


def upload_pyarrow_table_to_s3(
    table: pa.Table,
    s3_file_path: str,
) -> bool:
    """Uploads a PyArrow table to an S3 bucket as a Parquet file.

    Args:
        table: PyArrow Table to upload.
        s3_file_path: S3 key where the file will be stored. This is the complete path excluding
            the bucket name. It must end with .parquet.

    Returns:
        bool: True if the table was successfully uploaded, False otherwise.

    Raises:
        ValueError: If the provided path does not end with .parquet.
    """
    if not s3_file_path.endswith(".parquet"):
        err_msg = "File path must end with .parquet"
        logging.error(err_msg)
        raise ValueError(err_msg)

    # Convert PyArrow Table to Parquet and write to a BytesIO buffer
    buffer = BytesIO()
    pq.write_table(table=table, where=buffer)
    buffer.seek(0)

    # Create a session and resource
    s3 = s3io.get_s3_resource()
    bucket_name = app_state.bucket_name

    # Upload the file
    try:
        s3.Bucket(bucket_name).put_object(Key=s3_file_path, Body=buffer.getvalue())
        logging.info(f"Table successfully uploaded to s3://{bucket_name}/{s3_file_path}")
        return True
    except Exception as err:
        logging.error(f"Failed to upload table to S3: {err}")
        return False
