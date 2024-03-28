#!/usr/bin/env python
"""The article_segmentation batch job entry point.

Run with `python src/main.py -d us -c config_qa -p elasticity` from the project root.
"""
import logging

from elasticity.data import preprocessing
from elasticity.model.run_model import run_experiment_for_uids_parallel
from elasticity.utils import cli_default_args
from elasticity.utils.elasticity_action_list import generate_actions_list
from ql_toolkit.attrs.write import write_actions_list
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env import setup


def run() -> None:
    """This function is the entry point for the elasticity job.

    Returns:
        None
    """
    # Env Setup
    args_dict, config = setup.run_setup(args_dict=cli_default_args.args_kv)
    logging.info('args_dict: %s', args_dict)
    logging.info('config: %s', config)
    client_keys_map = config["client_keys"]
    # End of setup
    logging.info('bucket_name: %s', app_state.bucket_name)
    logging.info('client_keys_map: %s', client_keys_map)
    for client_key in client_keys_map:
        channels_list = client_keys_map[client_key]["channels"]
        for channel in channels_list:
            logging.info("Processing %s - %s", client_key, channel)
            df, total_end_date_uid = preprocessing.read_and_preprocess(client_key=client_key,
                                                   channel=channel,
                                                   price_changes=5, threshold=0.01,
                                                   min_days_with_conversions=10)
            logging.info("Total number of uid: %s", total_end_date_uid)

            # df = pd.read_csv('/home/alexia/workspace/elasticity/notebooks/
            # feeluniquecom_test_after_preprocessing.csv')
            # TODO: write df_results as report
            df_results = run_experiment_for_uids_parallel(df,
                                                          price_col='round_price',
                                                          quantity_col='units',
                                                          weights_col='days')
            logging.info('elasticity quality test: %s', df_results.quality_test.value_counts())
            actions_list = generate_actions_list(df_results, client_key, channel)

            write_actions_list(
                actions_list=actions_list,
                client_key=client_key,
                channel=channel,
                qa_run=False,
                is_local=True,
                filename_prefix=None,
                chunk_size=5000)
            logging.info("Finished processing %s - %s", client_key, channel)

if __name__ == "__main__":
    run()
