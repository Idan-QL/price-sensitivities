"""This module contains functions to retrieve the configuration dictionary from a YAML file."""

import logging
from typing import Optional

from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.s3 import io_tools as s3io


def retrieve_config(conf_file: str) -> dict:
    """This function retrieves the configuration dictionary from a YAML file from an S3 bucket.

    If the file is not found the function exits the program.

    Args:
        conf_file (str): The name of the configuration file.

    Returns:
        config (dict): The configuration dictionary.
    """
    config = s3io.maybe_get_yaml_file(s3_dir=app_state.s3_conf_dir, file_name=f"{conf_file}.yaml")
    if config is None:
        logging.error(f"Config file {conf_file} not found.")
    return config


def print_config(config: dict) -> None:
    """Print the configuration dictionary.

    Args:
        config (dict): The configuration dictionary to print

    Returns:
        None
    """
    logging.info("Printing Run Configuration:")
    print("----------")
    for key, val in config.items():
        print(f"{key}: {val} ({type(val)})")
    print("----------\n")


def is_containing_client_keys(config_dict: Optional[dict], client_keys: str) -> bool:
    """Check if the configuration contains client_keys.

    Args:
        config_dict (Optional[dict]): The configuration.
        client_keys (str): The client keys identifier.

    Returns:
        bool: True if the configuration contains client_keys.
    """
    return config_dict and config_dict.get(client_keys) is not None


def split_config_dict(config_dict: dict, client_keys_key: str) -> tuple[dict, dict]:
    """Break a configuration dictionary into two separate dictionaries.

    Args:
        config_dict (dict): The configuration dictionary.
        client_keys_key (str): The client-keys dictionary key in the config Dict.

    Returns:
        tuple[dict, dict]: A tuple containing two dictionaries:
            - One excluding the client_keys key.
            - One containing only the client_keys key.
    """
    # Dictionary excluding the client_key
    principal_configuration = {
        key: value for key, value in config_dict.items() if key != client_keys_key
    }

    # Dictionary containing only the client_key data, or an empty dictionary if not present
    client_keys_map = config_dict.get(client_keys_key, {})

    return principal_configuration, client_keys_map
