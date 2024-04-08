"""Module for writing attributes to a file."""

import json
import logging
from datetime import datetime
from os import makedirs, path

from ql_toolkit.attrs import action_list as al
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3 import io as s3io


def write_attributes(
    res_list: list,
    client_key: str,
    channel: str,
    attr_names: list[str],
    is_local: bool,
    qa_run: bool,
) -> None:
    """Create an actions list of results as attributes and write the list to an S3 directory.

    The attributes are written to a file in the S3 directory.

    Args:
        res_list (list): A list of iterables containing the results
        client_key (str): The client name
        channel (str): The channel
        attr_names (list[str]): The names of the attributes
        is_local (bool): Whether to write to local or s3
        qa_run (bool): Write results to a test directory

    Returns:
        None
    """
    logging.info("[- attrs -] Writing zero forecast results...")
    actions_list = al.create_actions_list(
        res_list=res_list, client_key=client_key, channel=channel, attr_names=attr_names
    )
    logging.info(
        "[- attrs -] Attributes created as an actions list. Writing to file..."
    )
    write_actions_list(
        actions_list=actions_list,
        client_key=client_key,
        channel=channel,
        filename_prefix="zero_forecasting_actions",
        is_local=is_local,
        qa_run=qa_run,
    )
    logging.info("[- attrs -] Attributes written!")


def write_actions_list(
    actions_list: list,
    client_key: str,
    channel: str,
    qa_run: bool,
    is_local: bool = False,
    filename_prefix: str = None,
    chunk_size: int = 5000,
) -> None:
    """Write actions list to file.

    The list is split into chunks of size n and each chunk is written to a separate file.
    n is hardcoded to 5000 for now.

    Args:
        actions_list (list): List of actions to be written to file
        client_key (str): The retailer name
        channel (str): The channel
        qa_run (bool): Write results to a test directory
        filename_prefix (str): The prefix of the file name. Defaults to None
        is_local (bool): Whether to write to local or s3. Defaults to False.
        chunk_size (int): The size of each chunk. Defaults to 5000.

    Returns:
        None
    """
    if filename_prefix is None:
        filename_prefix = f"{app_state.project_name}_actions"
    logging.info("Writing %s actions to file...", len(actions_list))

    for i in range(0, len(actions_list), chunk_size):
        chunk = actions_list[i : i + chunk_size]
        actions_str = "\n".join(map(lambda j: json.dumps(j), chunk))
        file_name = f"{filename_prefix}_{client_key}_{channel}_{i}_{datetime.now().isoformat()}.txt"
        monitor_run_dir = app_state.s3_monitoring_dir(
            client_key=client_key, channel=channel
        )
        if is_local:
            logging.info("[- attrs -] Writing files to local folder...")
            directory = "../artifacts/actions/"
            if not path.exists(directory):
                makedirs(directory)
            with open(path.join(directory, file_name), "w") as file:
                file.write(actions_str)
        else:
            s3_attrs_dir = (
                f"spark/output/test/{app_state.project_name}/"
                if qa_run
                else app_state.res_attrs_dir
            )
            logging.info("[- attrs -] Writing files to S3: %s", s3_attrs_dir)
            s3io.upload_to_s3(
                s3_dir=s3_attrs_dir, file_name=file_name, file_obj=actions_str
            )
            if not qa_run and not is_local:
                file_name = "_".join(file_name.split("_")[:-1]) + ".txt"
                s3io.upload_to_s3(
                    s3_dir=monitor_run_dir, file_name=file_name, file_obj=actions_str
                )
