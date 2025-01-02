"""Utilities for working with the DS Service API."""

import logging

from ql_toolkit.application_state.manager import app_state

SUPPORTED_RUN_ENVS = {"local", "staging", "production"}
SUPPORTED_DC_CODES = {"usea", "useb", "usecp", "euca", "local"}


def get_ds_service_api_url(port: int = 7393) -> str:
    """Get the URL for the DS Service API based on the current environment.

    Args:
        port (int, optional): The port number for the local API. Defaults to 7393.

    Returns:
        str: The URL for the DS Service API.
    """
    run_env = app_state.run_env
    dc_code = app_state.dc_code
    _validate_env_and_dc(run_env=run_env, dc_code=dc_code)

    ds_service_url_postfix = "ds-service/v1/"

    if app_state.run_env == "local":
        logging.info("Connecting to Local API & DB.")
        return f"http://127.0.0.1:{port}/{ds_service_url_postfix}"

    logging.info(f"Connecting to {run_env} database in {dc_code}...")

    return f"http://ds-service.{run_env}.{dc_code}.qldns.host:80/{ds_service_url_postfix}"


def _validate_env_and_dc(run_env: str, dc_code: str) -> None:
    if run_env not in SUPPORTED_RUN_ENVS:
        err_msg = f"Unsupported run environment: '{run_env}'. "
        logging.error(err_msg)
        raise ValueError(err_msg)

    if dc_code not in SUPPORTED_DC_CODES:
        err_msg = f"Unsupported dc_code: '{dc_code}'. "
        logging.error(err_msg)
        raise ValueError(err_msg)

    if run_env == "staging" and dc_code == "euca":
        err_msg = "The staging environment in euca is not supported."
        logging.error(err_msg)
        raise ValueError(err_msg)
