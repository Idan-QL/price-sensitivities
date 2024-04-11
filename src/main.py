#!/usr/bin/env python
"""The article_segmentation batch job entry point.

Run with `python src/main.py -d us -c config_qa -p elasticity` from the project root.
"""
import logging
from sys import exit as sys_exit
from datetime import datetime

import pandas as pd

from elasticity.data import preprocessing
from elasticity.model.run_model import run_experiment_for_uids_parallel
from elasticity.utils import cli_default_args
from elasticity.utils.elasticity_action_list import generate_actions_list
import elasticity.utils.plot_demands as plot_demands
from ql_toolkit.attrs.write import write_actions_list
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env import setup
from ql_toolkit.s3 import io as s3io
from report import report, write_graphs


def run() -> None:
    """This function is the entry point for the elasticity job.

    Returns:
        None
    """
    # Env Setup
    args_dict, config = setup.run_setup(args_dict=cli_default_args.args_kv)
    logging.info("args_dict: %s", args_dict)
    logging.info("config: %s", config)
    client_keys_map = config["client_keys"]
    # End of setup
    logging.info("bucket_name: %s", app_state.bucket_name)
    logging.info("client_keys_map: %s", client_keys_map)

    try:
        is_local = args_dict["local"]
        # Check if there is a "qa_run" key in the config
        # and if it is "true" or "True", set qa_run to True
        is_qa_run = config["qa_run"] if "qa_run" in config.keys() else False
    except KeyError as err:
        logging.error("KeyError: s", err)
        sys_exit("Exiting!")

    if is_local:
        logging.info(" ------ Running locally ------ ")
    elif is_qa_run:
        logging.info(" ------ Running in QA mode ------ ")
    else:
        logging.info(" ------ Running in Production mode ------ ")

    data_report = []
    for client_key in client_keys_map:
        channels_list = client_keys_map[client_key]["channels"]
        for channel in channels_list:
            try:
                logging.info("Processing %s - %s", client_key, channel)
                start_time = datetime.now()

                # df_by_price, _, total_end_date_uid, end_date = preprocessing.read_and_preprocess(
                #     client_key=client_key,
                #     channel=channel,
                #     price_changes=5,
                #     threshold=0.01,
                #     min_days_with_conversions=10,
                # )

                df_by_price = pd.read_csv('/home/alexia/workspace/elasticity/notebooks/feeluniquecom_test_after_preprocessing.csv')
                total_end_date_uid = 8074
                end_date = '2024-03-01'

                logging.info("End date: %s", end_date)
                logging.info("Total number of uid: %s", total_end_date_uid)

                df_results = run_experiment_for_uids_parallel(
                    df_by_price,
                    price_col="round_price",
                    quantity_col="units",
                    weights_col="days",
                )
                logging.info(
                    "elasticity quality test: %s",
                    df_results.quality_test.value_counts(),
                )

                s3io.write_dataframe_to_s3(
                    file_name=f"elasticity_{client_key}_{channel}_{end_date}.csv",
                    xdf=df_results,
                    s3_dir="data_science/eval_results/elasticity/",
                )

                #save top 10 graphs
                for uid in df_results[df_results["quality_test"] == True].sort_values('best_mean_relative_error')[:10]['uid'].unique():
                    buffer = plot_demands.plot_model_and_prices_buffer(df_results, df_by_price, uid)
                    s3io.upload_to_s3(s3_dir=f"data_science/eval_results/elasticity/graphs/{client_key}/{channel}/",
                      file_name=f"{uid}_{end_date}",
                      file_obj=buffer)

                actions_list = generate_actions_list(df_results, client_key, channel)

                write_actions_list(
                    actions_list=actions_list,
                    client_key=client_key,
                    channel=channel,
                    qa_run=True,
                    is_local=is_local,
                    filename_prefix=None,
                    chunk_size=5000,
                )

                runtime = (datetime.now() - start_time).total_seconds() / 60

                data_report = report.add_run(data_report=data_report,
                                             client_key=client_key,
                                             channel=channel,
                                             total_uid=total_end_date_uid,
                                             df_results=df_results,
                                             runtime=runtime)

                write_graphs.save_distribution_graph(client_key=client_key,
                                channel=channel,
                                total_uid=total_end_date_uid,
                                df_report=pd.DataFrame([data_report[-1]]),
                                end_date=end_date,
                                s3_dir="data_science/eval_results/elasticity/graphs/")

                logging.info("Finished processing %s - %s", client_key, channel)

            except Exception as e:
                logging.error("Error processing %s - %s: %s", client_key, channel, e)
                data_report = report.add_error_run(data_report=data_report,
                                                   client_key=client_key,
                                                   channel=channel)

    report_df = pd.DataFrame(data_report)

    s3io.write_dataframe_to_s3(
        file_name=f"elasticity_report_{end_date}.csv",
        xdf=report_df,
        s3_dir="data_science/eval_results/elasticity/",
    )
    


if __name__ == "__main__":
    run()
