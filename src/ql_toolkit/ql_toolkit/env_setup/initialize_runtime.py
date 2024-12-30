"""This module contains the function to set up the logging configuration."""

import logging
from typing import List

from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.s3.utils import set_aws_access
from ql_toolkit.env_setup.config_loader import spreadsheet_config, yaml_config
from ql_toolkit.env_setup.utils import cli, logs_config


def run_setup(
    cli_args_dict: dict,
    google_sheet_keep_cols: List[str],
    client_keys: str = "client_keys",
    channels: str = "channels",
) -> tuple[dict, dict]:
    """Set up the runtime environment.

    Args:
        cli_args_dict (dict): The CLI arguments dictionary
        google_sheet_keep_cols (List[str]): The columns to keep from the Google Sheet.
        client_keys (str): The client keys identifier. Default 'client_keys'.
        channels (str): The channels' identifier. Default 'channels'.

    Returns:
        tuple[dict, dict]: The config_dict and client_keys_map (from S3 or Google Sheet)
    """
    cli_args_dict = initial_environment_setup_by_cli(cli_args_dict)
    config_dict, client_keys_map = get_run_configuration(
        args_dict=cli_args_dict,
        google_sheet_keep_cols=google_sheet_keep_cols,
        client_keys_col_name=client_keys,
        channels_col_name=channels,
    )

    logging.info(f"config_dict: {config_dict}")
    logging.info(f"{app_state}")
    logging.info("Setup complete!")

    if app_state.results_type == "production":
        print(
            "\n\n\033[1;31m"  # Bold Red
            + "=" * 69
            + "\n\n"
            + "🚀🚀🚀 PRODUCTION RUN: The job is running in production mode! 🚀🚀🚀\n\n"
            + "=" * 69
            + "\033[0m\n\n"
        )

    return config_dict, client_keys_map


# Write a function to validate the args_dict has the required keys
def validate_args_dict(args_dict: dict) -> None:
    """Validate the args_dict has the required keys.

    Args:
        args_dict (dict): The CLI arguments dictionary

    Returns:
        None

    Raises:
        KeyError: If the required keys are not found in the args_dict
    """
    required_keys = [
        "aws_key",
        "aws_secret",
        "is_qa_run",
        "project_name",
        "storage_location",
    ]
    for key in required_keys:
        if key not in args_dict:
            raise KeyError(
                f"Key '{key}' not found in the args_dict. " f"Please add it to cli_default_args.py"
            )

    # Convert local flag to boolean if it's a string
    if isinstance(args_dict["is_qa_run"], str):
        args_dict["is_qa_run"] = args_dict["is_qa_run"].lower() == "true"


def initial_environment_setup_by_cli(cli_args_dict: dict) -> dict:
    """Set up the initial environment variables and logging by the CLI arguments.

    Args:
        cli_args_dict (dict): The CLI arguments dictionary

    Returns:
        dict: The updated CLI arguments
    """
    # Do not change the order of the following block of code
    # --- Start of the block
    # print here, because logging is not set up yet
    print("Setting up runtime environment...")
    # Keep the explicit args_dict value access to raise KeyError if the key is not found
    parser = cli.Parser(kv=cli_args_dict)
    parser.set_args_dict()
    cli_args_dict = parser.get_args_dict()
    validate_args_dict(args_dict=cli_args_dict)

    app_state.initialize(
        storage_location=cli_args_dict["storage_location"],
        project=cli_args_dict["project_name"],
        is_qa_run=cli_args_dict["is_qa_run"],
    )
    logs_config.set_up_logging()
    # --- End of the block

    # Convert local flag to boolean if it's a string
    if isinstance(cli_args_dict.get("local"), str):
        cli_args_dict["local"] = cli_args_dict["local"].lower() == "true"

    set_aws_access(aws_key=cli_args_dict["aws_key"], aws_secret=cli_args_dict["aws_secret"])
    return cli_args_dict


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
    config_dict = yaml_config.retrieve_config(conf_file=args_dict["config"])

    if yaml_config.is_containing_client_keys(
        config_dict=config_dict, client_keys=client_keys_col_name
    ):
        return yaml_config.split_config_dict(
            config_dict=config_dict, client_keys_key=client_keys_col_name
        )

    spreadsheet_name = spreadsheet_config.get_spreadsheet_name(config_dict=config_dict)
    client_keys_map = spreadsheet_config.fetch_google_sheet_config(
        spreadsheet_name=spreadsheet_name,
        google_sheet_keep_cols=google_sheet_keep_cols,
        client_keys_col_name=client_keys_col_name,
        channels_col_name=channels_col_name,
    )

    return config_dict, client_keys_map
