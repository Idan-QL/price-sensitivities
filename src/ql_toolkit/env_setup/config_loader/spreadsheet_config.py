"""This module contains functions to retrieve the configurations from a Google Sheet."""

import logging
from ast import literal_eval
from sys import exit as sys_exit
from typing import Any, Dict, List, Optional, Union

import gspread
import pandas as pd
from google.auth.exceptions import GoogleAuthError
from oauth2client.service_account import ServiceAccountCredentials

from ql_toolkit.application_state.manager import app_state
from ql_toolkit.aws_data_management.secrets_manager.loader import get_secret


class ConfigurationNotFoundError(Exception):
    """Exception raised for errors in the configuration retrieval process."""

    pass


def get_spreadsheet_name(config_dict: Optional[dict]) -> str:
    """Determine the spreadsheet name to use.

    Uses 'clients_sheet_name' from config_dict if available;
    otherwise, falls back to app_state.config_prod_spreadsheet.

    Args:
        config_dict (Optional[dict]): The S3 configuration.

    Returns:
        str: The name of the spreadsheet.
    """
    if config_dict and "clients_sheet_name" in config_dict:
        spreadsheet_name = config_dict["clients_sheet_name"]
        logging.error(f"Using spreadsheet name from config: {spreadsheet_name}")
    else:
        spreadsheet_name = app_state.config_prod_spreadsheet
        logging.warning(
            f"Config not found or 'clients_sheet_name' key missing. "
            f"Using default: {spreadsheet_name}",
        )

    return spreadsheet_name


def fetch_google_sheet_config(
    spreadsheet_name: str,
    google_sheet_keep_cols: List[str],
    client_keys_col_name: str,
    channels_col_name: str,
) -> dict:
    """Fetch and process configuration from Google Sheet.

    Args:
        spreadsheet_name (str): The name of the spreadsheet to read.
        google_sheet_keep_cols (List[str]): The columns to keep from the Google Sheet.
        client_keys_col_name (str): The client keys column name in the Google spreadsheet.
        channels_col_name (str): The channels' column name in the Google spreadsheet.

    Returns:
        dict: The configuration from Google Sheet.

    Raises:
        ConfigurationNotFoundError: If the configuration is not found or is corrupt.
    """
    try:
        google_sheet_config_dict = get_config_from_google_sheet(
            spreadsheet_name=spreadsheet_name,
            google_sheet_keep_cols=google_sheet_keep_cols,
            client_keys_col_name=client_keys_col_name,
            channels_col_name=channels_col_name,
        )
        if google_sheet_config_dict is None:
            msg = "Configuration from Google Sheet is None"
            logging.warning(msg)
            raise ValueError(msg)

        logging.info("Configuration successfully retrieved from Google Sheet.")
        return google_sheet_config_dict

    except Exception as google_sheet_error:
        logging.error(f"Error retrieving config from Google Sheet: {google_sheet_error}")
        msg = "Config not found or corrupt from both S3 and Google Sheet."
        raise ConfigurationNotFoundError(msg) from google_sheet_error


def get_config_from_google_sheet(
    spreadsheet_name: str,
    google_sheet_keep_cols: List[str],
    client_keys_col_name: str,
    channels_col_name: str,
) -> Dict[str, Dict]:
    """Retrieve and filter configuration from a Google Sheet based on data center.

    Args:
        spreadsheet_name (str): The name of the spreadsheet to read.
        google_sheet_keep_cols (List[str]): The columns to keep from the Google Sheet.
        client_keys_col_name (str): The client keys column name in the Google spreadsheet.
        channels_col_name (str): The channels' column name in the Google spreadsheet.

    Returns:
        Dict[str, Dict]: The configuration from Google Sheet. If None, the function exits.
    """
    sheet_name = f"{app_state.project_name}_clients_sheet"
    logging.info(f"Reading config: {spreadsheet_name}/{sheet_name}")
    try:
        google_sheet_df = get_google_sheet_data(
            spreadsheet_name=spreadsheet_name, sheet_name=sheet_name
        )

        filtered_df = filtered_google_sheet_by_region(
            google_sheet_client_keys_map=google_sheet_df, aws_region=app_state.aws_region
        )

        filtered_df = filter_google_sheet_columns(
            google_sheet_client_keys_map=filtered_df, google_sheet_keep_cols=google_sheet_keep_cols
        )

        if validate_config(filtered_df=filtered_df, client_keys=client_keys_col_name):
            clients_keys_map = transform_df_to_config_format(
                df=filtered_df,
                client_keys_col_name=client_keys_col_name,
                channels_col_name=channels_col_name,
            )
            if not clients_keys_map:
                logging.error("No configuration found in Google Sheet. Exiting application!")
                sys_exit()
            return clients_keys_map

    except Exception as err:
        if isinstance(err, ValueError):
            error_msg = f"Value error while reading config: {err}."
        else:
            error_msg = f"Unexpected error in get_config_from_google_sheet: {err}."
        error_msg += " Exiting application!"
        logging.error(error_msg)
        sys_exit()


def get_google_sheet_data(spreadsheet_name: str, sheet_name: str) -> pd.DataFrame:
    """Retrieve data from a Google Sheet and convert it to a pandas DataFrame.

    Args:
        spreadsheet_name (str): The name of the spreadsheet.
        sheet_name (str): The name of the sheet within the spreadsheet.

    Returns:
        pd.DataFrame: The data from the sheet as a pandas DataFrame.
    """
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    try:
        creds_dict = get_secret("prod/google_cloud/ds/api_key")

        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            keyfile_dict=creds_dict, scopes=scope
        )
        client = gspread.authorize(creds)
        sheet = client.open(spreadsheet_name).worksheet(sheet_name)
        data = sheet.get_all_records()

        if not data:
            sys_exit(f"No data found in sheet '{sheet_name}' of spreadsheet '{spreadsheet_name}'")

        return pd.DataFrame(data)

    except Exception as e:
        drive_config_exception(e)
        sys_exit(
            "Exit. Fatal error in get_google_sheet_data."
            f"Sheet: '{sheet_name}', spreadsheet: '{spreadsheet_name}'"
        )


def filter_google_sheet_columns(
    google_sheet_client_keys_map: pd.DataFrame, google_sheet_keep_cols: List[str]
) -> pd.DataFrame:
    """Filter the Google Sheet data by the specified columns.

    Args:
        google_sheet_client_keys_map (pd.DataFrame): The data from the sheet as a pandas DataFrame.
        google_sheet_keep_cols (List[str]): The columns to keep from the Google Sheet.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if not google_sheet_keep_cols:
        logging.warning("No columns specified to keep from the Google Sheet.")
        return google_sheet_client_keys_map
    return google_sheet_client_keys_map[google_sheet_keep_cols]


def derive_aws_region_code(aws_region: str) -> str:
    """Derive the AWS region code from the AWS region name.

    This function derives the AWS region code from the AWS region name by taking the first part
    of the region name before the hyphen and converting it to lowercase.

    Args:
        aws_region (str): The AWS region name.

    Returns:
        str: The AWS region code.

    Examples:
        >>> derive_aws_region_code("us-east-1")
        "us"
        >>> derive_aws_region_code("eu-central-1")
        "eu"
    """
    aws_region_code = aws_region.split("-")[0]
    return aws_region_code.lower()


def filtered_google_sheet_by_region(
    google_sheet_client_keys_map: pd.DataFrame, aws_region: str
) -> pd.DataFrame:
    """Get and filter the Google Sheet data by the AWS region.

    This function filters the Google Sheet data by the AWS region or data center, in order to get
    the configuration for the specified region.

    Args:
        google_sheet_client_keys_map (pd.DataFrame): The data from the sheet as a pandas DataFrame.
        aws_region (str): Data center to filter by.

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Raises:
        SystemExit: If no data is found for the specified data center.
    """
    filter_col = get_google_sheet_region_based_col(
        google_sheet_client_keys_map=google_sheet_client_keys_map
    )

    aws_region_code = derive_aws_region_code(aws_region)

    filtered_df = google_sheet_client_keys_map[
        google_sheet_client_keys_map[filter_col] == aws_region_code
    ]

    if not filtered_df.empty:
        return filtered_df.drop(filter_col, axis=1)

    err_msg = (
        f"No data found for requested region '{aws_region}' "
        f"(using region code: {aws_region_code})"
    )
    logging.error(err_msg)
    sys_exit(err_msg)


def get_google_sheet_region_based_col(google_sheet_client_keys_map: pd.DataFrame) -> str:
    """Get the column name to filter by region or data center.

    Args:
        google_sheet_client_keys_map (pd.DataFrame): The data from the sheet as a pandas DataFrame.

    Returns:
        str: The column name to filter by region or data center.
    """
    filter_col = "region" if "region" in google_sheet_client_keys_map.columns else "data_center"
    if filter_col not in google_sheet_client_keys_map.columns:
        logging.error(
            "No 'region' or 'data_center' column found in the Google Sheet. Exiting application!"
        )
        sys_exit()
    elif filter_col == "data_center":
        logging.warning("No 'region' column found. Defaulting to 'data_center'.")
    return filter_col


def validate_config(filtered_df: pd.DataFrame, client_keys: str) -> bool:
    """Validate config. Check for duplicates. Exits if errors are found.

    Args:
        filtered_df (pd.DataFrame): The DataFrame with the config filtered.
        client_keys (str): The client keys identifier.

    Returns:
        bool: True if tests passed.

    Raises:
        SystemExit: If duplicate entries are found or validation fails.
    """
    try:
        duplicated_keys_list = (
            filtered_df[client_keys].value_counts()[lambda x: x > 1].index.tolist()
        )
        if duplicated_keys_list:
            sys_exit(
                f"Spreadsheet config: Duplicate entries for client_keys: {duplicated_keys_list}"
            )
        return True
    except Exception as e:
        sys_exit(f"Exit following fatal error in validate_config: {e}")


def evaluate_list_value(value: str) -> Union[List, str]:
    """Attempt to evaluate a string as a list.

    This function attempts to evaluate a string as a list using the `literal_eval` function.
    If the evaluation fails, the original string is returned.

    Args:
        value (str): The value to evaluate.

    Returns:
        Union[List, str]: The evaluated list, or the original string if evaluation fails.
    """
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def evaluate_boolean_value(value: str) -> Optional[bool]:
    """Convert a string to a boolean - if possible.

    This function converts a string to a boolean if the string represents a boolean value.
    The function returns None if the string does not represent a boolean value.

    Args:
        value (str): The value to process.

    Returns:
        Optional[bool]: The processed boolean value, or None if the value is not a boolean.
    """
    value_lower = value.lower()
    if value_lower == "true":
        return True
    if value_lower == "false":
        return False
    return None


def evaluate_string_value(value: str) -> Optional[Union[str, List, bool]]:
    """Process string values, handling None, lists, and booleans.

    This function processes string values, handling the following cases:

    - If the string is empty or represents "None", "null", "unk", or "unknown", the function
        returns `None`.
    - If the string represents a list (e.g., "[1, 2, 3]"), the function attempts to evaluate
        the string and convert it into an actual Python list.
    - If the string represents a boolean ("True", "False", "true", "false"), the function
        converts it to the corresponding boolean value (`True` or `False`).
    - If the string cannot be evaluated or does not match any specific patterns, the original
        string is returned.

    Args:
        value (str): The value to process.

    Returns:
        Optional[Union[str, List, bool]]: The processed value.
    """
    if value.lower() in {"", "none", "null", "unk", "unknown"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return evaluate_list_value(value=value)
    bool_value = evaluate_boolean_value(value=value)
    if bool_value is not None:
        return bool_value
    return value


def evaluate_value(
    value: Union[str, int, float, bool, List]
) -> Optional[Union[str, int, float, bool, List]]:
    """Processes a single value from a DataFrame row, handling specific cases for strings.

    This function is designed to process individual cell values in a DataFrame row. It handles
    specific cases for strings, such as converting empty strings to `None`, evaluating strings
    representing lists, and converting strings representing boolean values to actual booleans.
    If the value is not a string, it is returned as is.

    Args:
        value (Union[str, int, float, List]): The value to process, which may be a string,
                                              integer, float, or list.

    Returns:
        Optional[Union[str, int, float, List]]: The processed value, which may be `None`,
            a string, an integer, a float, or a list. If the input value is an empty string or
            "None", the function returns `None`. If the value is a string representing a list,
            the function returns the evaluated list. Otherwise, the original value is returned.
    """
    if isinstance(value, str):
        return evaluate_string_value(value=value)
    return value


def convert_row_attributes_to_dict(
    row: pd.Series, client_keys_col_name: str, channels_col_name: str
) -> Dict[str, Any]:
    """Extract and process attributes from a single DataFrame row.

    This function processes each column in the given DataFrame row, excluding the
    `client_keys` and `channels` columns. It converts string representations of lists
    to actual lists and handles empty or "None" strings appropriately.

    Args:
        row (pd.Series): The row to process, containing various attributes as columns.
        client_keys_col_name (str): The column name identifying client keys, which will be excluded
            from processing.
        channels_col_name (str): The column name identifying channels, which will be excluded
            from processing.

    Returns:
        Dict[str, Any]: A dictionary containing the processed attributes, where the key is the
            column name and the value is the processed data from that column.
    """
    attributes = {}
    for column, value in row.items():
        if column not in {client_keys_col_name, channels_col_name}:
            attributes[str(column)] = evaluate_value(value)

    return attributes


def transform_df_to_config_format(
    df: pd.DataFrame, client_keys_col_name: str, channels_col_name: str
) -> Dict[str, Dict]:
    """Convert a DataFrame into a configuration dictionary format.

    This function iterates through each row of the DataFrame, converting specific columns
    into a dictionary format required for further processing or configuration.
    The `client_keys_col_name` is used as the key in the resulting dictionary, and other columns
    are processed and added as attributes.

    Args:
        df (pd.DataFrame): The DataFrame to transform, containing the data to be converted.
        client_keys_col_name (str): The column name identifying client keys, which will be used as
            keys in the resulting dictionary.
        channels_col_name (str): The column name identifying channels, which will be processed and
            added as attributes.

    Returns:
        Dict[str, Dict]: A dict where each key is a client key, and the value is a dictionary of
                        attributes including the channels data and other processed row attributes.
    """
    config = {}

    for _, row in df.iterrows():
        client_keys_value = row.get(client_keys_col_name)
        channels_value = row.get(channels_col_name)

        # Convert the channels column value to a list if it's a list-like string
        if (
            isinstance(channels_value, str)
            and channels_value.startswith("[")
            and channels_value.endswith("]")
        ):
            try:
                channels_value = literal_eval(channels_value)
            except (ValueError, SyntaxError):
                logging.warning(f"Could not evaluate channels value as list: {channels_value}")

        # Extract other attributes from the row
        attributes = convert_row_attributes_to_dict(
            row=row, client_keys_col_name=client_keys_col_name, channels_col_name=channels_col_name
        )

        # Store the configuration, with client keys as the key, to be consistent with the
        # YAML configuration format
        config[client_keys_value] = {channels_col_name: channels_value, **attributes}

    return config


def drive_config_exception(e: Exception) -> None:
    """Handle exceptions and log appropriate warnings.

    Args:
        e (Exception): The exception to handle.
    """
    if isinstance(e, (gspread.exceptions.GSpreadException, gspread.exceptions.APIError)):
        logging.error(f"GSpreadException: {e}")
    elif isinstance(
        e, (gspread.exceptions.SpreadsheetNotFound, gspread.exceptions.WorksheetNotFound)
    ):
        logging.error(f"Spreadsheet/Worksheet error: {e}")
    elif isinstance(e, GoogleAuthError):
        logging.warning(f"GoogleAuthError: {e}")
    elif isinstance(e, ValueError):
        logging.warning(f"Value error while reading config: {e}")
    else:
        logging.warning(f"Unexpected error: {e}")
