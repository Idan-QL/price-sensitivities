"""This module contains functions to retrieve and print the configuration dictionary."""

import logging
from os import path
from sys import exit as sys_exit

import yaml

from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3 import io as s3io


def get_args_dict(is_local: bool, conf_file: str) -> dict:
    """This function retrieves the configuration dictionary from a YAML file.

    If the file is local, it checks if the file exists and then loads the YAML content.
    If the file is not local, it retrieves the file from an S3 bucket.
    If the file is not found in either case, the function exits the program.

    Args:
        is_local (bool): A flag indicating whether the file is local or not.
        conf_file (str): The name of the configuration file.

    Returns:
        config (dict): The configuration dictionary.
    """
    if is_local:
        if not path.exists(conf_file):
            sys_exit(
                f"Config file {conf_file} not found.\n"
                f"Please check the command line file name provided.\nExiting!"
            )
        try:
            with open(file=conf_file, mode="r", encoding=None) as file_stream:
                config = yaml.safe_load(stream=file_stream)
        except FileNotFoundError:
            sys_exit(
                f"Config file {conf_file} not found. "
                f"Please check the command line file name provided.\nExiting!"
            )
    else:
        config = s3io.maybe_get_yaml_file(
            s3_dir=app_state.s3_conf_dir, file_name=f"{conf_file}.yaml"
        )
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
