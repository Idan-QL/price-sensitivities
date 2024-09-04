"""Module of utils.utils."""

import logging


def log_environment_mode(is_local: bool, is_qa_run: bool) -> None:
    """Log the current environment mode based on the provided flags.

    Args:
        is_local (bool): Flag indicating if running locally.
        is_qa_run (bool): Flag indicating if running in QA mode.
    """
    if is_local:
        logging.info(" ------ Running locally ------ ")
    if is_qa_run:
        logging.info(" ------ Running in QA mode ------ ")
    if not is_local and not is_qa_run:
        logging.info(" ------ Running in Production mode ------ ")


def run_type(is_local: bool, is_qa_run: bool) -> str:
    """Return the run type.

    Args:
        is_local (bool): Flag indicating if running locally.
        is_qa_run (bool): Flag indicating if running in QA mode.

    Returns:
        str: The type of run, which can be 'local', 'qa', or 'production'.
    """
    if is_local:
        return "local"
    if is_qa_run:
        return "qa"
    return "production"
