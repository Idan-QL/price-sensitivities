#!/usr/bin/env python
"""The article_segmentation batch job entry point.

Run with `python src/main.py -d us -c config_qa -p elasticity` from the project root.
"""
import logging
from datetime import datetime

from ql_toolkit.runtime_env import setup
from elasticity.utils import cli_default_args
from elasticity.data import preprocessing
from elasticity.model.run_model import run_experiment_for_uids_parallel
import pandas as pd

from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.attrs.action_list import create_actions_list
from ql_toolkit.attrs.write import write_actions_list
import numpy as np

def run() -> None:
    """This function is the entry point for the article_segmentation batch job.

    Returns:
        None
    """
    # Env Setup
    args_dict, config = setup.run_setup(args_dict=cli_default_args.args_kv)
    print(args_dict)
    print(config)
    client_keys_map = config["client_keys"]
    # End of setup

    print(app_state.s3_db_sets_dir)
    print(app_state.bucket_name)

    print(config)
    print(client_keys_map)

    for client_key in client_keys_map:
        channels_list = client_keys_map[client_key]["channels"]
        for channel in channels_list:
            logging.info(f"Processing {client_key} - {channel}")
            # df = preprocessing.read_and_preprocess(client_key=client_key,
            #                                        channel=channel,
            #                                        price_changes=5, threshold=0.01,
            #                                        min_days_with_conversions=10)
            
            df = pd.read_csv('/home/alexia/workspace/elasticity/notebooks/feeluniquecom_test_after_preprocessing.csv')
            #write df_results as report
            df_results = run_experiment_for_uids_parallel(df, price_col='round_price', quantity_col='units', weights_col='days')
            print(df_results.quality_test.value_counts())
            
            attr_cs = [
                "uid",
                "best_model_a", # best_model_a
                "best_model_b", # best_model_b
                "best_model", # best_model
                "power_elasticity", #power_elasticity (or best_model_elasticity)
                ]
            
            df_actions = df_results[df_results.quality_test][attr_cs]
            df_actions['qlia_elasticity_calc_date'] = datetime.now().strftime(app_state.date_format)

            attr_names = [
                "uid",
                "qlia_elasticity_param1",
                "qlia_elasticity_param2",
                "qlia_elasticity_demand_model",
                "qlia_product_elasticity",
                "qlia_elasticity_calc_date",
                ]
            
            res_list = [tuple(row) for row in df_actions.values]

            print(len(res_list))

            actions_list = create_actions_list(
                res_list=res_list,
                client_key="client_key",
                channel="channel",
                attr_names=attr_names
                )

            write_actions_list(
                actions_list=actions_list,
                client_key=client_key,
                channel=channel,
                qa_run=False,
                is_local=True,
                filename_prefix=None,
                chunk_size=5000)


if __name__ == "__main__":
    run()
