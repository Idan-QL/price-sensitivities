"""This module contains the function to set up the logging configuration."""

import logging
from datetime import datetime, timezone
from os import makedirs, path
from sys import stdout

from ql_toolkit.application_state.manager import app_state


def set_up_logging(is_debug: bool = False, logs_dir_path: str = "") -> None:
    """Set up the logging configuration with immediate flush."""
    proj_name = app_state.project_name
    if not logs_dir_path:
        logs_dir_path = f"./logs/{proj_name}_{datetime.now(tz=timezone.utc):%Y-%m-%dT%H_%M_%S}"

    # Ensure directory exists
    makedirs(name=logs_dir_path, exist_ok=True)

    log_file_name = path.join(logs_dir_path, "run_stdout.log")

    # Create a logger and clear existing handlers
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define handlers with immediate flush
    file_handler = logging.FileHandler(log_file_name, delay=False)
    file_handler.setLevel(logging.DEBUG)  # Ensure all levels are logged to file
    stream_handler = logging.StreamHandler(stdout)

    # Set logging level
    logging_level = logging.DEBUG if is_debug else logging.INFO
    logger.setLevel(logging_level)

    # Define format and attach handlers
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s]: %(message)s (%(filename)s/%(funcName)s)",
        datefmt=f"{app_state.date_format} %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Ensure file handler flushes immediately
    file_handler.flush()

    # Log a test message to confirm setup
    logger.info("Logging has been set up successfully.")
