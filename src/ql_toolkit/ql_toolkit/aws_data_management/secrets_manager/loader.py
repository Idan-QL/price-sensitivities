"""This module contains the functions to interact with AWS Secrets Manager."""

import json
import logging

import boto3
from botocore.exceptions import ClientError

from ql_toolkit.application_state.manager import app_state


def parse_secret(secret: str) -> dict:
    """Parse the secret string into a dictionary.

    Args:
        secret (str): The secret string to parse.

    Returns:
        dict: The parsed secret.
    """
    return json.loads(secret)


def get_secret(secret_name: str) -> dict:
    """Get the secret from AWS Secrets Manager.

    The `get_secret_value` function retrieves the secret value from AWS Secrets Manager as a
    JSON string. This string is then parsed into a dictionary using the `parse_secret` function.

    Args:
        secret_name (str): The name of the secret to retrieve. For example,
            "rds/internal-attributes-production-euca/rw".

    Returns:
        dict: The parsed secret string.
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=app_state.aws_region)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as err:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        logging.error(f"Failed to get secret value: {err}")
        raise err

    secret = get_secret_value_response["SecretString"]
    return parse_secret(secret=secret)
