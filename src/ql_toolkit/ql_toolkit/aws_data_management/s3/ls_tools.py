"""This module contains functions to list files and directories in an S3 bucket."""

import logging
from os import path

import boto3
from botocore.exceptions import ClientError
from botocore.paginate import PageIterator

from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.s3.utils import get_s3_client


def is_file_exists(s3_dir: str, file_name: str, s3_client: boto3.client = None) -> bool:
    """Check whether file_path exists in S3.

    Raises Error is unexpected ClientError was encountered.

    Args:
        s3_dir (str): The path to the directory (without s3://<bucket>/)
        file_name (str): The name of the file to check
        s3_client (boto3.client): The S3 client object

    Returns:
        (bool): True if the file exists
    """
    if s3_client is None:
        s3_client = get_s3_client()
    file_path = path.join(s3_dir, file_name)
    try:
        s3_client.head_object(Bucket=app_state.bucket_name, Key=file_path)
        return True
    except ClientError as err:
        if err.response["Error"]["Code"] == "404":
            logging.error(f"File not found: {file_path}")
        elif err.response["Error"]["Code"] == "403":
            logging.error(f"Unauthorized request to get file: {file_path}")
        else:
            logging.error(f"Unknown error when reading file: {file_path}")
            raise err
        return False


def list_dir(s3_dir: str, delimiter: str = "/", s3_client: boto3.client = None) -> dict:
    """Lists the objects in an S3 bucket with a specific directory path and delimiter.

    It checks if an S3 client is provided; if not, it obtains the client.

    Args:
        delimiter (str, default "/"): Used to restrict the result to only include
            application_state prefixes (directories) and not individual objects
        s3_dir (str): The prefix (directory) to be listed
        s3_client (boto3.client): The S3 client

    Returns:
        (dict):
    """
    if s3_client is None:
        s3_client = get_s3_client()
    if not s3_dir.endswith("/"):
        s3_dir += "/"
    return s3_client.list_objects_v2(
        Bucket=app_state.bucket_name, Prefix=s3_dir, Delimiter=delimiter
    )


def get_common_dir_prefixes(
    s3_dir: str, delimiter: str = "/", s3_client: boto3.client = None
) -> list[str]:
    """Get the list of directories directly inside dir_path.

    This function does not get a recursive list of directories.
    The CommonPrefixes key in the list_dir response contains a list of dictionaries,
    each representing a directory.
    By iterating over these directories, we can access the Prefix key to get the directory name.
    The returned list gets rid of the dicts and returns just the directories paths.
        delimiter (str, default "/"): Used to restrict the result to only include
            application_state prefixes (directories) and not individual objects
        dir_path (str): The prefix (directory) to be listed
        s3_client (boto3.client):

    Returns:
        (list[str]): A list of strings, each representing a directory in dir_path
    """
    objects = list_dir(s3_dir=s3_dir, delimiter=delimiter, s3_client=s3_client)
    prefixes_list = objects.get("CommonPrefixes", [])
    return [x["Prefix"] for x in prefixes_list]


def list_dir_pages(s3_dir: str, s3_client: boto3.client = None) -> PageIterator:
    """List all the objects in an S3 directory, using pagination.

    Args:
        s3_dir (str): The S3 directory path
        s3_client (boto3.client): The S3 client

    Returns:
         botocore.paginate.PageIterator: An iterator over the pages of the list
    """
    if s3_client is None:
        s3_client = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")
    return paginator.paginate(Bucket=app_state.bucket_name, Prefix=s3_dir, FetchOwner=False)


def list_files_in_dir(s3_dir: str, file_type: str = "json", s3_client: boto3.client = None) -> set:
    """Get an unsorted *set* of files of type "file_type" from an S3 directory.

    Args:
        s3_dir (str): The S3 directory path
        file_type (str): The of files to list
        s3_client (boto3.client): The S3 client

    Returns:
        set: A set of file_type files in dir_path
    """
    pages = list_dir_pages(s3_dir, s3_client)
    ret = []
    for page in pages:
        if "Contents" in page:
            ret += [
                c["Key"].split(file_type)[0] + file_type
                for c in page["Contents"]
                if file_type in c["Key"]
            ]

    return set(ret)


def folder_contents_exist(s3_dir: str) -> bool:
    """This function checks if a folder contains files (returns True) or not (returns False).

    Args:
        s3_dir (str): The path to the directory to check

    Returns:
        (bool): True when the folder contains files, else False
    """
    bucket_name = app_state.bucket_name
    s3_client = get_s3_client()
    if not s3_dir.endswith("/"):
        s3_dir += "/"
    resp = s3_client.list_objects(Bucket=bucket_name, Prefix=s3_dir, Delimiter="/", MaxKeys=1)
    return "Contents" in resp or "CommonPrefixes" in resp
