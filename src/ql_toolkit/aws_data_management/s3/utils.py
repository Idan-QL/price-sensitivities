"""Utility functions for interacting with AWS S3."""

from os import environ
from typing import Optional

import boto3


def set_aws_access(aws_key: str, aws_secret: str) -> None:
    """This function is used to set the AWS access keys and data center.

    Args:
       aws_key (str): The AWS access key ID. This should be a string of alphanumeric characters.
       aws_secret (str): The AWS secret access key.
                         This should be a string of alphanumeric characters.

    Returns:
       None
    """
    if len(aws_key) > 0:
        environ["AWS_ACCESS_KEY_ID"] = aws_key
    if len(aws_key) > 0:
        environ["AWS_SECRET_ACCESS_KEY"] = aws_secret


def get_aws_access_key() -> Optional[str]:
    """Get the AWS access key from the environment variables.

    Returns:
        Optional[str]: The AWS access key
    """
    return environ.get("AWS_ACCESS_KEY_ID")


def get_aws_secret_key() -> Optional[str]:
    """Get the AWS secret key from the environment variables.

    Returns:
        Optional[str]: The AWS secret key
    """
    return environ.get("AWS_SECRET_ACCESS_KEY")


def get_s3_resource() -> boto3.resource:
    """This function is used to get an AWS S3 resource object.

    It first retrieves the AWS access key and secret key from the environment variables.
    If either the access key or secret key is not set, it returns a default S3 resource object.
    If both keys are set, it returns an S3 resource object initialized with these keys.

    Returns:
        boto3.resources.factory.s3.ServiceResource: The AWS S3 resource object.
    """
    access_key = get_aws_access_key()
    secret_key = get_aws_secret_key()
    if not access_key or not secret_key:
        return boto3.resource("s3")
    return boto3.resource("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)


def get_s3_client() -> boto3.client:
    """This function is used to get an AWS S3 client object.

    It first retrieves the AWS access key and secret key from the environment variables.
    If either the access key or secret key is not set, it returns a default S3 client object.
    If both keys are set, it returns an S3 client object initialized with these keys.

    Returns:
        boto3.client: The AWS S3 client object.
    """
    access_key = get_aws_access_key()
    secret_key = get_aws_secret_key()
    if not access_key or not secret_key:
        return boto3.client("s3")
    return boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)
