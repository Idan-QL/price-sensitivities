"""This module contains functions to write attributes to a file."""

import json
import logging
from datetime import datetime, timezone
from os import path

from pydantic import ValidationError

from ql_toolkit.attrs import action_list as al
from ql_toolkit.attrs import data_classes as dc
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.s3 import s3io


def write_attributes(
    res_list: list,
    attr_names: list[str],
    **kwargs: any,
) -> None:
    """Create an actions list of results as attributes and write the list to an S3 directory.

    The attributes are written to a file in the S3 directory.

    Args:
        res_list (list): A list of iterables containing the results
        attr_names (list[str]): The names of the attributes
        **kwargs: Variable length keyword arguments, expected to contain:
            - client_key (str): The client name
            - channel (str): The channel
            - is_local (bool): Whether to write to local or S3
            - qa_run (bool): Write results to a test directory
    Returns:
        None
    """
    try:
        input_data = dc.WriteAttributesKWArgs(**kwargs)
    except ValidationError as err:
        logging.error(f"ValidationError caught: {err}")
        raise

    logging.info("[- attrs -] Writing results...")
    actions_list = al.create_actions_list(
        res_list=res_list,
        client_key=input_data.client_key,
        channel=input_data.channel,
        attr_names=attr_names,
    )
    logging.info("[- attrs -] Attributes created as an actions list. Writing to file...")

    input_args = dc.WriteActionsListKWArgs(
        client_key=input_data.client_key,
        channel=input_data.channel,
        is_local=input_data.is_local,
        qa_run=input_data.qa_run,
    )

    _write_actions_list(
        actions_list=actions_list,
        input_args=input_args,
    )
    logging.info("[- attrs -] Attributes written!")


def _write_actions_list(
    actions_list: list,
    input_args: dc.WriteActionsListKWArgs,
) -> None:
    """Write actions list to file.

    The list is split into chunks of size n and each chunk is written to a separate file.
    n is hardcoded to 5000 for now.

    Args:
        actions_list (list): List of actions to be written to file
        input_args (WriteActionsListKWArgs): Keyword arguments for writing the actions list
            - client_key (str): The client name
            - channel (str): The channel
            - is_local (bool, Default False): Whether to write to local or S3
            - qa_run (bool): Write results to a test directory
            - filename_prefix (str, Default None): The prefix of the file name
            - chunk_size (int, Default 5000): The size of the chunks

    Returns:
        None
    """
    filename_prefix = input_args.filename_prefix
    if filename_prefix is None:
        filename_prefix = f"{app_state.project_name}_actions"
    logging.info("Writing %s actions to file...", len(actions_list))
    # monitor_run_dir = app_state.s3_monitoring_dir(client_key=client_key, channel=channel)

    for i in range(0, len(actions_list), input_args.chunk_size):
        chunk = actions_list[i : i + input_args.chunk_size]
        actions_str = "\n".join(json.dumps(j) for j in chunk)
        file_name = (
            f"{filename_prefix}_{input_args.client_key}_{input_args.channel}_{i}"
            f"_{datetime.now(tz=timezone.utc).isoformat()}.txt"
        )
        if input_args.is_local:
            logging.info("[- attrs -] Writing files to local folder...")
            with open(
                file=path.join("../artifacts/actions/", file_name),
                mode="w",
                encoding=None,
            ) as file:
                file.write(actions_str)
        else:
            s3_attrs_dir = (
                f"spark/output/test/{app_state.project_name}/"
                if input_args.qa_run
                else app_state.res_attrs_dir
            )
            logging.info(f"[- attrs -] Writing files to S3: {path.join(s3_attrs_dir, file_name)}")
            s3io.upload_to_s3(s3_dir=s3_attrs_dir, file_name=file_name, file_obj=actions_str)
            # if not qa_run and not is_local:
            #     file_name = "_".join(file_name.split("_")[:-1]) + ".txt"
            #     s3io.upload_to_s3(
            #         s3_dir=monitor_run_dir,
            #         file_name=file_name,
            #         file_obj=actions_str)
