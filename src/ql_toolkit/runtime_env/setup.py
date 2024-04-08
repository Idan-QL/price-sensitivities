"""This module contains the function to set up the logging configuration."""

import logging
from sys import exit as sys_exit

from ql_toolkit.config import external_config
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env.utils import cli, logs_config
from ql_toolkit.s3.utils import set_aws_access


def run_setup(args_dict: dict) -> tuple[dict, dict]:
    """This function sets up the runtime environment for the job."""
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
    s3_config = external_config.get_args_dict(
        is_local=False, conf_file=args_dict["config"]
    )
    if s3_config is None:
        sys_exit(f"Config file {args_dict['config']} not found.\nExiting!")
    # parse_config.print_config(config)

    logging.info("Setup complete!")
    return args_dict, s3_config
