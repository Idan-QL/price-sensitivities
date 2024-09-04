"""This module contains functions to retrieve and print the configuration dictionary."""

import logging

from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3 import io_tools as s3io


def get_yaml_config(conf_file: str) -> dict:
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
