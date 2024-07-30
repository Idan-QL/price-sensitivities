"""This module contains the function to set up the logging configuration."""

import logging
from ast import literal_eval
from sys import exit as sys_exit

import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

from ql_toolkit.config import external_config
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env.utils import cli, logs_config
from ql_toolkit.s3.io import maybe_get_json_file
from ql_toolkit.s3.utils import set_aws_access


def get_google_sheet_data(spreadsheet_name: str, sheet_name: str, creds_dict: dict) -> pd.DataFrame:
    """Retrieve data from a Google Sheet and convert it to a pandas DataFrame.

    Args:
        spreadsheet_name (str): The name of the spreadsheet.
        sheet_name (str): The name of the sheet within the spreadsheet.
        creds_dict (dict): The Google API credentials.

    Returns:
        pd.DataFrame: The data from the sheet as a pandas DataFrame.
    """
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(spreadsheet_name).worksheet(sheet_name)
    return pd.DataFrame(sheet.get_all_records())


def get_config_from_sheet(spreadsheet_name: str, sheet_name: str, data_center: str) -> dict:
    """Retrieve and filter configuration from a Google Sheet based on data center.

    Args:
        spreadsheet_name (str): The name of the spreadsheet to read.
        sheet_name (str): The name of the sheet to read.
        data_center (str): The data center to filter by.

    Returns:
        dict: The filtered configuration in JSON format.
    """
    logging.info(f"Reading config: {spreadsheet_name}/{sheet_name}/{data_center}")
    try:
        # Download the credentials file
        creds_dict = maybe_get_json_file(
            s3_dir=app_state.s3_config_directory, file_name="ds_config_credential.json"
        )

        # Get the sheet into a DataFrame
        df_google_sheet = get_google_sheet_data(spreadsheet_name, sheet_name, creds_dict)
        # Filter the DataFrame based on the data center
        df_filtered = df_google_sheet[df_google_sheet["data_center"] == data_center]

        # Check for multiple lines for the same client_key
        duplicated_keys = df_filtered["client_key"].value_counts()[lambda x: x > 1].index.tolist()
        if duplicated_keys:
            sys_exit(
                f"Spreadsheet config: Duplicate entries found for client_keys: {duplicated_keys}"
            )

        # Transform the filtered DataFrame to the required JSON format
        config = {}
        for _, row in df_filtered.iterrows():
            client_key = row["client_key"]
            # Parse channels as a list
            channels = (
                literal_eval(row["channels"])
                if isinstance(row["channels"], str)
                else [row["channels"]]
            )
            attr_name = (
                [row["attributes"]]
                if not isinstance(row["attributes"], list)
                else row["attributes"]
            )
            if (attr_name == [""]) or (attr_name == ["None"]):
                attr_name = [None]

            config[client_key] = {"channels": channels, "attr_name": attr_name}

        return {"client_keys": config}

    except gspread.exceptions.GSpreadException as e:
        logging.warning(f"Google Sheets API error: {e}")
    except ValueError as e:
        logging.warning(f"Value error while reading config: {e}")
    except Exception as e:
        logging.warning(f"Unexpected error : {e}")
    return None


def run_setup(args_dict: dict) -> tuple[dict, dict]:
    """Set up the runtime environment.

    Args:
        args_dict (dict): The CLI arguments

    Returns:
        tuple[dict, dict]: The CLI arguments and the configuration (from S3 or Google Sheet)
    """
    print("Setting up runtime environment...")
    parser = cli.Parser(kv=args_dict)
    parser.set_args_dict()
    args_dict = parser.get_args_dict()
    app_state.project_name = args_dict["project_name"]
    app_state.bucket_name = args_dict["data_center"]
    logs_config.set_up_logging()
    logging.info("Setting up runtime environment...")

    # Validate the boolean CLI args
    if not isinstance(args_dict["local"], bool):
        args_dict["local"] = args_dict["local"].lower() == "true"

    print("Job CLI args:")
    for k, v in args_dict.items():
        if "aws" in k:
            continue
        print(f"{k}: {v} ({type(v)})")

    # We should set the AWS credentials as environment variables to ensure the
    # S3 resource initialization is possible from everywhere in the script
    set_aws_access(aws_key=args_dict["aws_key"], aws_secret=args_dict["aws_secret"])
    logging.info(
        "Running on the %s Data Center (%s Bucket)",
        args_dict["data_center"].upper(),
        app_state.bucket_name,
    )

    # Get the run configuration
    s3_config = external_config.get_args_dict(is_local=False, conf_file=args_dict["config"])
    if s3_config is None:
        logging.warning(
            f"Config file {args_dict['config']} not found in S3. Falling back to Google Sheet."
        )

        # Define environment to spreadsheet mapping
        config_spreadsheet_map = {
            True: app_state.config_qa_spreadsheet,  # QA Environment
            False: app_state.config_prod_spreadsheet,  # Production Environment
        }
        is_qa_run = args_dict.get("is_qa_run", False)
        spreadsheet_name = config_spreadsheet_map.get(is_qa_run)
        s3_config = get_config_from_sheet(
            spreadsheet_name, app_state.project_name, args_dict["data_center"]
        )
        if s3_config is None:
            sys_exit("Config not found or corrupt.\nExiting!")

    logging.info("Setup complete!")
    return args_dict, s3_config
