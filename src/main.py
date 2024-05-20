#!/usr/bin/env python
"""The article_segmentation batch job entry point.

Run with `python src/main.py -d us -c config_qa -p elasticity` from the project root.
"""
import logging
import traceback
from datetime import datetime
from sys import exit as sys_exit

import pandas as pd

import elasticity.utils.plot_demands as plot_demands
from elasticity.data import preprocessing
from elasticity.model.run_model import run_experiment_for_uids_parallel
from elasticity.utils import cli_default_args
from elasticity.utils.elasticity_action_list import generate_actions_list
from ql_toolkit.attrs.write import write_actions_list
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env import setup
from ql_toolkit.s3 import io as s3io
from report import logging_error, report, write_graphs


def setup_environment() -> tuple:
    """Set up the environment and return args and config.

    Returns:
        tuple: args_dict, config, client_keys_map, is_local, is_qa_run
    """
    args_dict, config = setup.run_setup(args_dict=cli_default_args.args_kv)
    logging.info("args_dict: %s", args_dict)
    logging.info("config: %s", config)
    client_keys_map = config["client_keys"]
    logging.info(f"client_keys_map: {client_keys_map}")

    try:
        is_local = args_dict["local"]
        is_qa_run = config.get("qa_run", False)
    except KeyError as err:
        logging.error(f"KeyError: {err}")
        sys_exit("Exiting!")

    if is_local:
        logging.info(" ------ Running locally ------ ")
    elif is_qa_run:
        logging.info(" ------ Running in QA mode ------ ")
    else:
        logging.info(" ------ Running in Production mode ------ ")

    return args_dict, config, client_keys_map, is_local, is_qa_run


def process_client_channel(
    data_report: list, client_key: str, channel: str, is_local: bool
) -> dict:
    """Process a client and channel and return the results.

    Args:
        data_report (list): List of the data report by client/channel
        client_key (str): The client key.
        channel (str): The channel.
        is_local (bool): Flag indicating if the script is running locally.

    Returns:
        dict: The results of processing.
    """
    error_counter = logging_error.ErrorCounter()
    logging.getLogger().addHandler(error_counter)
    try:
        logging.info(f"Processing {client_key} - {channel}")
        start_time = datetime.now()

        df_by_price, _, total_end_date_uid, end_date, df_revenue_uid, total_revenue = (
            preprocessing.read_and_preprocess(client_key=client_key, channel=channel)
        )

        logging.info(f"End date: {end_date}")
        logging.info(f"Total number of uid: {total_end_date_uid}")
        logging.info(f"total_revenue: {total_revenue}")
        outliers_count = df_by_price[df_by_price["outlier_quantity"]].uid.nunique()
        logging.info(f"Number of uid with outliers: {outliers_count}")

        df_results = run_experiment_for_uids_parallel(
            df_by_price[~df_by_price["outlier_quantity"]],
            price_col="round_price",
            quantity_col="units",
            weights_col="days",
        )

        df_results = df_results.merge(df_revenue_uid, on="uid", how="left")

        logging.info(
            f"elasticity quality test: {df_results.quality_test.value_counts()}"
        )
        logging.info(
            f"elasticity quality test high: {df_results.quality_test_high.value_counts()}"
        )

        s3io.write_dataframe_to_s3(
            file_name=f"elasticity_{client_key}_{channel}_{end_date}.csv",
            xdf=df_results,
            s3_dir="data_science/eval_results/elasticity/",
        )

        plot_demands.run_save_graph_top10(
            df_results, df_by_price, client_key, channel, end_date
        )

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

        data_report = report.add_run(
            data_report=data_report,
            client_key=client_key,
            channel=channel,
            total_uid=total_end_date_uid,
            df_results=df_results,
            total_revenue=total_revenue,
            runtime=runtime,
            error_counter=error_counter.error_count,
            end_date=end_date,
        )

        write_graphs.save_distribution_graph(
            client_key=client_key,
            channel=channel,
            total_uid=total_end_date_uid,
            df_report=pd.DataFrame([data_report[-1]]),
            end_date=end_date,
            s3_dir=app_state.s3_eval_results_dir() + "graphs/",
        )

    except (KeyError, pd.errors.EmptyDataError) as e:
        logging.error(f"Error processing {client_key} - {channel}: {e}")
        error_info = traceback.format_exc()
        logging.error(f"Error occurred in {__file__} - {e} \n{error_info}")
        data_report = report.add_error_run(
            data_report=data_report,
            client_key=client_key,
            channel=channel,
            error_counter=error_counter.error_count,
        )
    return data_report


def run() -> None:
    """Main function to run the elasticity job."""
    args_dict, config, client_keys_map, is_local, is_qa_run = setup_environment()
    data_report = []

    for client_key in client_keys_map:
        channels_list = client_keys_map[client_key]["channels"]
        for channel in channels_list:
            data_report = process_client_channel(
                data_report, client_key, channel, is_local
            )

    report_df = pd.DataFrame(data_report)

    end_date = datetime.now().strftime("%Y-%m-%d")
    s3io.write_dataframe_to_s3(
        file_name=f"elasticity_report_{args_dict['config']}_{end_date}.csv",
        xdf=report_df,
        s3_dir=app_state.s3_eval_results_dir(),
    )


if __name__ == "__main__":
    run()
