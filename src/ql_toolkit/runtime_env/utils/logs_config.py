"""This module contains the function to set up the logging configuration."""

import logging
from datetime import datetime
from os import makedirs, path
from sys import stdout

from ql_toolkit.config.runtime_config import app_state


def set_up_logging(is_debug: bool = False, logs_dir_path: str = "") -> None:
    """Set up the logging configuration.

    Args:
        is_debug (bool): Whether to set the logging level to DEBUG. Defaults to False.
        logs_dir_path (str): The path to the logs directory. Defaults to None.

    Returns:
        None
    """
    # Set logging
    # noinspection PyUnusedLocal
    # logger = logging.getLogger(__name__)
    proj_name = app_state.project_name
    if not logs_dir_path:
        logs_dir_path = f"./logs/{proj_name}_{datetime.now():%Y-%m-%dT%H_%M_%S}"
    makedirs(logs_dir_path)
    log_file_name = path.join(logs_dir_path, "run_stdout.log")
    handlers_list = [logging.FileHandler(log_file_name)]
    logging_level = logging.DEBUG if is_debug else logging.INFO
    handlers_list.append(logging.StreamHandler(stdout))
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s] [%(levelname)-8s]: %(message)s (%(filename)s/%(funcName)s)",
        datefmt=f"{app_state.date_format} %H:%M:%S",
        handlers=handlers_list,
    )
    # return logger
