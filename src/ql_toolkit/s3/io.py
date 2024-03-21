"""This module contains functions for reading and writing files to and from an Amazon S3 bucket."""

import json
import logging
from io import BytesIO, StringIO
from os import path
from tempfile import TemporaryFile
from typing import Optional, Union

import boto3
import pandas as pd
import polars as pl
import yaml
from botocore.exceptions import ClientError
from joblib import dump
from pyarrow.lib import ArrowInvalid
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3.ls import is_file_exists
from ql_toolkit.s3.utils import get_s3_client, get_s3_resource


def download_file(
    s3_dir: str,
    file_name: str,
    local_path: str,
    s3_rsc: boto3.resource = None,
) -> None:
    """Downloads a file from an S3 bucket and saves it directly to a specified local path.

    This function is useful when one wants to store the file on disk, for example,
    when dealing with large files or when the file is required for later use.

    Args:
        s3_dir (str): The S3 directory where the file is located.
        file_name (str): The name of the file to be downloaded.
        local_path (str): The local path where the file will be saved.
        s3_rsc (boto3.resource, optional): An S3 resource object.
            If not provided, a new S3 resource object will be created.

    Returns:
        None
    """
    file_name = path.join(s3_dir, file_name)
    if s3_rsc is None:
        s3_rsc = get_s3_resource()
    s3_rsc.Bucket(app_state.bucket_name).download_file(file_name, local_path)


def get_s3_file_contents(
    s3_dir: str,
    file_name: str,
    decode_utf8: bool,
    s3_rsc: boto3.resource = None,
) -> Union[BytesIO, str, None]:
    """Retrieves a file from an S3 bucket and returns its content.

    The content is returned either as a BytesIO object or a UTF-8 string,
    instead of saving it to disk.

    A BytesIO object is an in-memory stream for binary data. It allows you to work with the
    downloaded content
    directly in memory, which can be useful for smaller files or when you need to process the
    data immediately without storing it.
    This approach is beneficial if you need to pass the file content to another function or
    process without
    the overhead of file I/O operations on the disk.
    Use decode_utf8=True when trying to read, a .txt or a .yaml file, for example.
    Further processing may be required to convert the string to a dictionary or a list. For example:
        yaml.safe_load(file_content)
    or to a list:
        file_content.splitlines()

    Args:
        s3_dir (str): The S3 directory (without bucket name,
            e.g., 'forecasting/branch/default/ml_datasets/')
        file_name (str): The specific file name (e.g., 'data.parquet')
        s3_rsc (boto3.resource, optional): An S3 resource object. I
            f not provided, a new S3 resource object will be created.
        decode_utf8 (bool): If True, attempt to decode the file content as a UTF-8 string.

    Returns:
        Union[BytesIO, str, None]: A BytesIO object if `decode_utf8` is False, a UTF-8 string if
        the decoding is successful, or None if an error occurs.
    """
    if s3_rsc is None:
        s3_rsc = get_s3_resource()

    file_path = path.join(s3_dir, file_name)
    obj = s3_rsc.Object(app_state.bucket_name, file_path)
    buffer = BytesIO()

    try:
        obj.download_fileobj(buffer)
        if decode_utf8:
            try:
                # Reset buffer position to the start
                buffer.seek(0)
                return buffer.read().decode("utf-8")
            except UnicodeDecodeError as err:
                logging.warning("[- S3 I/O -] UTF-8 decoding failed for %s: %s", file_name, err)
                # Return as binary if UTF-8 decoding fails
                buffer.seek(0)
        return buffer
    except ClientError as err:
        logging.warning("[- S3 I/O -] Error caught while downloading %s: %s", file_name, err)
        return None


def maybe_get_python_objects_from_file(s3_dir: str, file_name: str) -> dict:
    """Get a dictionary of python objects from a python (.py) file.

    Args:
        s3_dir (str): The S3 directory that should hold the file
                      (For example: "data_science/sagemaker/clients_mappings/")
        file_name (str): The name of the file
                         (For example: "clients_dict.py")

    Returns:
        (dict) The file contents
    """
    file_content = get_s3_file_contents(s3_dir=s3_dir, file_name=file_name, decode_utf8=True)
    py_objects_map = {}
    if file_content is not None:
        exec(file_content, py_objects_map)
    return py_objects_map


def maybe_get_json_file(
    s3_dir: str,
    file_name: str,
    s3_rsc: boto3.resource = None,
) -> dict:
    """Get a dictionary with the retailer name and their associated channels as a list.

    The forecasting will be done for each channel of each retailer name (client key)

    Args:
        s3_dir (str): The S3 directory that should hold the file
        file_name (str): The name of the file
        s3_rsc (boto3.resource): An S3 resource object

    Raises:
        json.decoder.JSONDecodeError: If the json file cannot be decoded
        TypeError: If the json file is None

    Returns:
        (dict): The content of the json file. Empty dict if the file cannot be read.
                The user is responsible for checking if the returned value is valid,
                    based on its type.
    """
    content = get_s3_file_contents(
        s3_dir=s3_dir, file_name=file_name, decode_utf8=True, s3_rsc=s3_rsc
    )
    try:
        return json.loads(content)
    except json.decoder.JSONDecodeError as err:
        logging.error(
            "[- S3 I/O -] Error caught while reading the client_keys.json file:\n%s",
            err,
        )
        return {}
    except TypeError as err:
        logging.error("[- S3 I/O -] Error caught: %s", err)
        return {}


def maybe_get_yaml_file(
    s3_dir: str,
    file_name: str,
    s3_rsc: boto3.resource = None,
) -> Optional[Union[dict, list]]:
    """Read a YAML file, if it exists, from S3 and return it as a Python object (dict or list).

    Args:
        s3_dir (str): The S3 directory (without bucket name, e.g., 'configurations/default/')
        file_name (str): The specific file name (e.g., 'config.yaml')
        s3_rsc (boto3.resource, optional): An S3 resource object.

    Returns:
        Optional[Union[dict, list]]: A Python object (dict or list) if the file is found and
            successfully read, None otherwise.
    """
    file_content = get_s3_file_contents(
        s3_dir=s3_dir, file_name=file_name, s3_rsc=s3_rsc, decode_utf8=True
    )

    if file_content and isinstance(file_content, str):
        try:
            return yaml.safe_load(StringIO(file_content))
        except yaml.YAMLError as err:
            logging.error("[- S3 I/O -] Error parsing YAML file %s: %s", file_name, err)
        except Exception as err:
            logging.error(
                "[- S3 I/O -] Unexpected error while reading YAML file %s: %s",
                file_name,
                err,
            )
    else:
        logging.warning(
            "[- S3 I/O -] File %s not found or could not be decoded as UTF-8!",
            file_name,
        )

    return None


def maybe_get_txt_file(
    s3_dir: str,
    file_name: str,
    s3_rsc: boto3.resource = None,
) -> str:
    """Read a text file, if it exists, from S3 and return its content as a string.

    Args:
        s3_dir (str): The S3 directory (without bucket name, e.g., 'logs/default/')
        file_name (str): The specific file name (e.g., 'error_log.txt')
        s3_rsc (boto3.resource, optional): An S3 resource object.

    Returns:
        str: The content of the text file as a string if the file is found and
                       successfully read, otherwise an empty string is returned.
                       The user is responsible for checking if the returned value is valid,
                       based on its type.
    """
    file_content = get_s3_file_contents(
        s3_dir=s3_dir, file_name=file_name, s3_rsc=s3_rsc, decode_utf8=True
    )

    if file_content and isinstance(file_content, str):
        return file_content
    logging.error(f"[- S3 I/O -] File {file_name} not found or could not be decoded as UTF-8!")

    return ""


def maybe_get_pd_csv_df(
    s3_dir: str,
    file_name: str,
    parse_dates: Union[bool, list] = False,
    s3_rsc: boto3.resource = None,
) -> pd.DataFrame:
    """Read a CSV file, if it exists, from S3 and return it as a pandas DataFrame.

    Remove the infamous "Unnamed: 0" column, if it exists.

    Args:
        s3_dir (str): The S3 directory
            (without bucket name, e.g., 'forecasting/branch/default/ml_datasets/')
        file_name (str): The specific file name (e.g., 'data.csv')
        parse_dates (Union[bool, list[str]]): if True, try parsing the index as datetime Index.
            If a list of int or str is given, try parsing columns - each as a separate date column.
        s3_rsc (boto3.resource, optional): An S3 resource object.

    Returns:
        pd.DataFrame: A pandas DataFrame.
                      If the file is not found or not successfully read,
                      an empty pd.DataFrame is returned.
                      The user is responsible for checking if the returned value is valid,
                      based on its type.
    """
    file_content = get_s3_file_contents(
        s3_dir=s3_dir, file_name=file_name, s3_rsc=s3_rsc, decode_utf8=True
    )

    if file_content and isinstance(file_content, str):
        csv_df = pd.read_csv(StringIO(file_content), parse_dates=parse_dates)
        if not csv_df.empty:
            if "Unnamed: 0" in csv_df.columns:
                csv_df = csv_df.drop(columns="Unnamed: 0")
            return csv_df
        logging.warning("[- S3 I/O -] File %s is empty!", file_name)
    else:
        logging.warning(
            "[- S3 I/O -] File %s not found or could not be decoded as UTF-8!",
            file_name,
        )

    return pd.DataFrame()


def maybe_get_pd_parquet_file(
    s3_dir: str,
    file_name: str,
    s3_rsc: boto3.resource = None,
    validate_datetime_index: bool = False,
) -> pd.DataFrame:
    """Get a parquet file from S3 and optionally validate if it has a datetime index.

    Args:
        s3_dir (str): The S3 directory
            (without bucket name, e.g., 'forecasting/branch/default/ml_datasets/')
        file_name (str): The specific file name (e.g., 'tft_dataset.parquet')
        s3_rsc (boto3.resource, optional): An S3 resource object
        validate_datetime_index (bool, default False): If True, validates and sets the dataframe
            index to a datetime index if it's not already set.

    Returns:
        pd.DataFrame: A pandas DataFrame.
                      If the file is not found or not successfully read,
                      an empty pd.DataFrame is returned.
                      The user is responsible for checking if the returned value is valid,
                      based on its type.
    """
    buffer = get_s3_file_contents(
        s3_dir=s3_dir, file_name=file_name, decode_utf8=False, s3_rsc=s3_rsc
    )

    if buffer is None:
        logging.warning("[- S3 I/O -] Failed to retrieve file %s from S3.", file_name)
        return pd.DataFrame()

    try:
        df = pd.read_parquet(BytesIO(buffer.getvalue()))
    except Exception as err:
        logging.error("[- S3 I/O -] Error reading parquet file %s: %s", file_name, err)
        return pd.DataFrame()

    if validate_datetime_index:
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                try:
                    df = df.set_index(pd.to_datetime(df["date"]), drop=True)
                except Exception as err:
                    logging.warning(
                        "[- S3 I/O -] Failed to convert 'date' column to "
                        "datetime index in %s: %s",
                        file_name,
                        err,
                    )
            else:
                logging.warning(
                    "[- S3 I/O -] 'date' column not found in %s to set as datetime index.",
                    file_name,
                )
    return df


def write_dataframe_to_s3(
    file_name: str,
    xdf: pl.DataFrame | pd.DataFrame,
    s3_dir: Optional[str] = None,
    rename_old: bool = True,
    backup_path: str = "",
) -> None:
    """Write a DataFrame to S3 as a parquet or CSV file, based on the file extension in file_name.

    The path to the file is determined by the `s3_dir` parameter. If `s3_dir` is not provided,
    then `file_name` is used as the full path to the file.
    Otherwise, the full path is determined by joining `s3_dir` and `file_name`.

    Args:
        s3_dir (Optional[str]): The S3 directory that should hold the file. Default is None.
        file_name (str): The complete path to the file, e.g., "ml/sktime_dataset.parquet"
        or "ml/data.csv".
        xdf (Union[pl.DataFrame, pd.DataFrame]): The DataFrame to save to S3.
        rename_old (bool): If True, rename the existing file as a backup.
        backup_path (str, optional): The path to save the backup file.
            If None, a default path is used.

    Returns:
        None
    """
    if isinstance(xdf, pl.DataFrame):
        if xdf.is_empty():
            logging.warning("[- S3 I/O -] Provided DataFrame is empty. Aborting write operation.")
            return
    else:
        if xdf.empty:
            logging.warning("[- S3 I/O -] Provided DataFrame is empty. Aborting write operation.")
            return

    if not file_name.endswith((".parquet", ".csv")):
        logging.error("[- S3 I/O -] File extension must be .parquet or .csv")
        return

    s3_client = get_s3_client()
    bucket_name = app_state.bucket_name
    file_path = _get_file_path(file_name=file_name, s3_dir=s3_dir)

    # Backup existing file
    if rename_old and is_file_exists(s3_dir=s3_dir, file_name=file_name, s3_client=s3_client):
        _create_backup(
            backup_path=backup_path,
            bucket_name=bucket_name,
            file_path=file_path,
            s3_client=s3_client,
        )
    else:
        logging.warning("[- S3 I/O -] Skipping renaming of existing file!")

    # Write DataFrame to buffer
    buffer = BytesIO()
    if isinstance(xdf, pd.DataFrame):
        xdf = pl.from_pandas(xdf)

    try:
        if file_name.endswith(".parquet"):
            xdf.write_parquet(buffer)
        elif file_name.endswith(".csv"):
            xdf.write_csv(buffer)

        buffer.seek(0)  # Reset buffer position to the beginning
        s3_client.put_object(Bucket=bucket_name, Key=file_path, Body=buffer.getvalue())
        logging.info("[- S3 I/O -] %s successfully uploaded to S3.", file_path)
    except ArrowInvalid as err:
        logging.error("[- S3 I/O -] ArrowInvalid error caught: %s", err)
    except Exception as err:  # Catch more general exception for robustness
        logging.error("[- S3 I/O -] Error occurred while writing DataFrame to S3: %s", err)
        return


def _get_file_path(file_name: str, s3_dir: Optional[str]) -> str:
    """Constructs the full file path for a file in an S3 directory.

    This function takes a file name and an optional S3 directory as input and returns the full path
    to the file.
    If `s3_dir` is not provided, the function returns `file_name` as is.
    Otherwise, it joins `s3_dir` and `file_name` to form the full path.

    Args:
        file_name (str): The name of the file.
        s3_dir (Optional[str]): The S3 directory where the file is located.

    Returns:
        str: The full path to the file.
    """
    if not file_name and not s3_dir:
        raise ValueError("Either file_name or s3_dir must be provided!")
    return file_name if not s3_dir else path.join(s3_dir, file_name)


def _create_backup(
    backup_path: str, bucket_name: str, file_path: str, s3_client: boto3.client
) -> None:
    """Creates a backup of an existing file in an S3 bucket.

    This function copies the existing file to a new location within the same S3 bucket.
    The new location is determined by the `backup_path` parameter. If `backup_path` is not provided,
    the function will create a new path by inserting the string "backups" before the file name in
    the original path.

    Args:
        backup_path (str): The path where the backup file should be stored.
            If not provided, a default path is used.
        bucket_name (str): The name of the S3 bucket where the file is located.
        file_path (str): The path of the file to be backed up.
        s3_client (boto3.client): A boto3 S3 client object.

    Returns:
        None
    """
    backup_file_path = (
        backup_path
        or
        # Insert the string "backup" before the file name
        f"{file_path.rsplit(sep='/', maxsplit=1)[0]}/backups/"
        f"{file_path.rsplit(sep='/', maxsplit=1)[-1]}"
    )
    s3_client.copy_object(
        Bucket=bucket_name,
        CopySource={"Bucket": bucket_name, "Key": file_path},
        Key=backup_file_path,
    )
    logging.info("[- S3 I/O -] Backup of existing file saved to %s", backup_file_path)


def upload_to_s3(
    s3_dir: str,
    file_name: str,
    file_obj: any,
    is_serializable: bool = False,
    extra_args: Optional[dict] = None,
    s3_client: boto3.client = None,
    s3_rsc: boto3.resource = None,
) -> bool:
    """Upload a file object or a serializable object to an Amazon S3 bucket.

    This function can handle both file-like objects and objects that need to be serialized.
    For serializable objects, it serializes and writes them to the specified S3 directory and
    file name.
    For file-like objects, it uploads them directly.

    If the file object is serializable (i.e. it is a dict, list, set or a str) or is flagged as
    `is_serializable`,
    the function will attempt to serialize it and upload the serialized object to S3.
    Otherwise, it will upload the file-like object directly.

    `extra_args` can be used to pass additional arguments to the boto3 upload function. For example,
    to enable server-side encryption, you can pass `extra_args={"ServerSideEncryption": "AES256"}`.

    Args:
        s3_dir (str): The S3 directory where the file should be stored.
        file_name (str): The name to assign to the file in S3.
        file_obj (Any): The object to be uploaded.
            Can be a file-like object or a serializable object.
        is_serializable (bool, optional): If True, the object will be serialized before upload.
            Default is False.
        extra_args (dict, optional): Additional arguments for the boto3 upload function.
        s3_client (boto3.client, optional): A boto3 S3 client object.
            If not provided, a new client will be created.
        s3_rsc (boto3.resource, optional): An S3 resource object.
            Only needed for serializable objects.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    bucket_name = app_state.bucket_name
    file_path = _get_file_path(file_name=file_name, s3_dir=s3_dir)

    try:
        # If the file object is a string, encode it as bytes and upload directly
        if isinstance(file_obj, str):
            if s3_client is None:
                s3_client = get_s3_client()
            bytes_str = BytesIO(file_obj.encode())
            s3_client.upload_fileobj(
                Fileobj=bytes_str,
                Bucket=bucket_name,
                Key=file_path,
                ExtraArgs=extra_args,
            )

        # If the file object is a serializable object, serialize it and upload to S3
        elif is_serializable or isinstance(file_obj, (dict, list, set)):
            if s3_rsc is None and is_serializable:
                s3_rsc = boto3.resource("s3")
            # Serialize the object and put in S3
            with TemporaryFile() as buffer:
                dump(
                    value=file_obj, filename=buffer
                )  # Replace with appropriate serialization function
                buffer.seek(0)
                s3_rsc.Object(bucket_name, file_path).put(Body=buffer.read())
        # Otherwise, upload the file-like object directly
        else:
            if s3_client is None:
                s3_client = boto3.client("s3")
            # Upload the file-like object directly
            s3_client.upload_fileobj(
                Fileobj=file_obj,
                Bucket=bucket_name,
                Key=file_path,
                ExtraArgs=extra_args,
            )

        logging.info("[- S3 I/O -] Object uploaded to: s3://%s/%s", bucket_name, file_path)
        return True
    except Exception as err:
        logging.error("[- S3 I/O -] Error occurred while uploading to S3: %s", err)
        return False


def upload_local_file_to_s3(
    local_file_name: str,
    s3_file_name: str,
    s3_dir: Optional[str] = None,
) -> bool:
    """Upload a local file to S3.

    Args:
        s3_file_name (str): The file name on S3
        s3_dir (Optional[str]): The directory name on S3. If None, the file_name is used as the
            full path (without bucket name)
        local_file_name (Optional[str]): The COMPLETE path to the file name on the local machine

    Returns:
        (bool)
    """
    bucket_name = app_state.bucket_name
    s3_client = get_s3_client()
    s3_file_path = _get_file_path(file_name=s3_file_name, s3_dir=s3_dir)
    try:
        _ = s3_client.upload_file(local_file_name, bucket_name, s3_file_path)
    except ClientError as err:
        logging.error(f"[- S3 I/O -] Error uploading file to S3: {err}")
        return False
    return True
