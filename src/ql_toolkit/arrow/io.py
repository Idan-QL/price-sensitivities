"""This module contains functions for reading and writing Arrow tables using Polars DataFrames."""

import logging
from typing import Optional

import polars as pl
import pyarrow.parquet as pq
import s3fs
from pyarrow import Table

from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3 import io as s3io


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
        See:
            https://arrow.apache.org/docs/python/generated/
            pyarrow.parquet.ParquetDataset.html#pyarrow.parquet.ParquetDataset

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
    # if path_or_paths is a list, then add the bucket name to each path
    if isinstance(path_or_paths, list):
        path_or_paths = [f"{app_state.bucket_name}/{path}" for path in path_or_paths]
    else:
        path_or_paths = f"{app_state.bucket_name}/{path_or_paths}"
    try:
        pq_ds = pq.ParquetDataset(
            path_or_paths=path_or_paths, filesystem=s3_fs, filters=filters
        )
        table = pq_ds.read(columns=columns)
    except ValueError as err:
        if isinstance(path_or_paths, list):
            logging.error("ValueError caught: %s for the paths list", err)
        else:
            logging.error(
                "ValueError caught: %s for %s", err, path_or_paths[0].split("/")[-1]
            )
    except Exception as err:
        logging.error("Err caught: %s", err)
    return table


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

    Args:
        path_or_paths (str or list[str]): The path or list of paths to the Parquet file(s) to read.
        is_lazy (bool): Specifies whether to return a LazyFrame (True) or DataFrame (False).
        s3_fs (s3fs.S3FileSystem, default None): The S3FileSystem to use to read Parquet file(s).
        columns (list[str], default None): The list of columns to read from the Parquet file(s).
        filters (list[tuple], default None): The list of filters to apply to the Parquet file(s).
        See:
            https://arrow.apache.org/docs/python/generated/
            pyarrow.parquet.ParquetDataset.html#pyarrow.parquet.ParquetDataset

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
