#!/usr/bin/env python
"""The article_segmentation batch job entry point.

Run with `python src/main.py -d us -c config_qa -p elasticity` from the project root.
"""
import logging

from ql_toolkit.runtime_env import setup
from elasticity.utils import cli_default_args
from elasticity.data import preprocessing

from ql_toolkit.config.runtime_config import app_state


def run() -> None:
    """This function is the entry point for the article_segmentation batch job.

    Returns:
        None
    """
    # Env Setup
    args_dict, config = setup.run_setup(args_dict=cli_default_args.args_kv)
    client_keys_map = config["client_keys"]
    # End of setup

    print(config)
    print(client_keys_map)

    for client_key in client_keys_map:
        channels_list = client_keys_map[client_key]["channels"]
        for channel in channels_list:
            logging.info(f"Processing {client_key} - {channel}")

            print(app_state.s3_db_sets_dir)
            print(app_state.bucket_name)

            df = preprocessing.read_and_preprocess(client_key,channel, "X",
                                                   price_changes=5, threshold=0.01,
                                                   min_days_with_conversions=10,
                                                   start_date='2023-03-01', end_date='2024-02-29')


if __name__ == "__main__":
    run()
