"""This module contains the function to set up the logging configuration."""

import logging
from typing import List

from ql_toolkit.config import external_config
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.config.utils import (
    fetch_google_sheet_config,
    get_spreadsheet_name,
    is_client_keys_config,
    split_config_dict,
)
from ql_toolkit.runtime_env.utils import cli, logs_config
from ql_toolkit.s3.utils import set_aws_access


def run_setup(
    args_dict: dict,
    google_sheet_keep_cols: List[str],
    client_keys: str = "client_keys",
    channels: str = "channels",
) -> tuple[dict, dict]:
    """Set up the runtime environment.

    Args:
        args_dict (dict): The CLI arguments dictionary
        google_sheet_keep_cols (List[str]): The columns to keep from the Google Sheet.
        client_keys (str): The client keys identifier. Default 'client_keys'.
        channels (str): The channels' identifier. Default 'channels'.

    Returns:
        tuple[dict, dict]: The config_dict and client_keys_map (from S3 or Google Sheet)
    """
    print("Setting up runtime environment...")

    args_dict = setup_initial_environment(args_dict)
    config_dict, client_keys_map = get_run_configuration(
        args_dict,
        google_sheet_keep_cols=google_sheet_keep_cols,
        client_keys_col_name=client_keys,
        channels_col_name=channels,
    )
    # add local to config_dict
    config_dict["local"] = args_dict.get("local", False)
    logging.info("Setup complete!")
    return config_dict, client_keys_map


def setup_initial_environment(args_dict: dict) -> dict:
    """Set up the initial environment variables and logging.

    Args:
        args_dict (dict): The CLI arguments

    Returns:
        dict: The updated CLI arguments
    """
    parser = cli.Parser(kv=args_dict)
    parser.set_args_dict()
    args_dict = parser.get_args_dict()
    app_state.project_name = args_dict["project_name"]
    app_state.bucket_name = args_dict["data_center"]
    logs_config.set_up_logging()

    # Convert local flag to boolean if it's a string
    if isinstance(args_dict["local"], str):
        args_dict["local"] = args_dict["local"].lower() == "true"

    set_aws_access(aws_key=args_dict["aws_key"], aws_secret=args_dict["aws_secret"])
    logging.info(
        "Running on the %s Data Center (%s Bucket)",
        args_dict["data_center"],
        app_state.bucket_name,
    )
    return args_dict


def get_run_configuration(
    args_dict: dict,
    google_sheet_keep_cols: List[str],
    client_keys_col_name: str,
    channels_col_name: str,
) -> tuple[dict, dict]:
    """Get the run configuration from S3 or Google Sheet.

    Args:
        args_dict (dict): The CLI arguments
        google_sheet_keep_cols (List[str]): The columns to keep from the Google Sheet.
        client_keys_col_name (str): The client keys column name in the Google spreadsheet.
        channels_col_name (str): The channels' column name in the Google spreadsheet.

    Returns:
        tuple[dict, dict]: The config_dict and client_keys_map (from S3 or Google Sheet)

    Raises:
        ConfigurationNotFoundError: If the configuration is not found or is corrupt
    """
    config_dict = external_config.get_yaml_config(conf_file=args_dict["config"])

    if is_client_keys_config(config_dict=config_dict, client_keys=client_keys_col_name):
        return split_config_dict(config_dict=config_dict, client_keys_key=client_keys_col_name)

    spreadsheet_name = get_spreadsheet_name(config_dict=config_dict)
    client_keys_map = fetch_google_sheet_config(
        spreadsheet_name=spreadsheet_name,
        data_center=args_dict["data_center"],
        google_sheet_keep_cols=google_sheet_keep_cols,
        client_keys_col_name=client_keys_col_name,
        channels_col_name=channels_col_name,
    )

    return config_dict, client_keys_map
