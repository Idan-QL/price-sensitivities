#!/usr/bin/env python
"""The elasticity batch job entry point.

source $(poetry env info --path)/bin/activate
Run with `python src/main.py -d us -c config_qa -p elasticity` from the project root.
"""
import logging
import traceback
from datetime import datetime

import pandas as pd

import elasticity.utils.plot_demands as plot_demands
from elasticity.data import preprocessing
from elasticity.data.group import data_for_group_elasticity
from elasticity.data.utils import initialize_dates
from elasticity.model.group import add_group_elasticity
from elasticity.model.run_model import run_experiment_for_uids_parallel
from elasticity.utils import cli_default_args
from elasticity.utils.elasticity_action_list import generate_actions_list
from elasticity.utils.utils import log_environment_mode
from elasticity.utils.write import upload_elasticity_data_to_athena
from ql_toolkit.attrs import data_classes as dc
from ql_toolkit.attrs.write import _write_actions_list
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env import setup
from ql_toolkit.s3 import io_tools as s3io
from report import logging_error, report, write_graphs

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def setup_environment() -> tuple:
    """Set up the environment and return args and config.

    Returns:
        tuple: args_dict, client_keys_map, is_local, is_qa_run
    """
    config_dict, client_keys_map = setup.run_setup(
        args_dict=cli_default_args.args_kv,
        google_sheet_keep_cols=["client_keys", "channels", "attr_name"],
    )
    logging.info("config_dict: %s", config_dict)
    logging.info("client_keys_map: %s", client_keys_map)

    start_date, end_date = initialize_dates()

    is_local = config_dict.get("local", False)
    is_qa_run = config_dict.get("is_qa_run", False)
    log_environment_mode(is_local=is_local, is_qa_run=is_qa_run)

    return (
        client_keys_map,
        is_local,
        is_qa_run,
        start_date,
        end_date,
    )


def process_client_channel(
    client_key: str,
    channel: str,
    attr_name: str,
    is_local: bool,
    is_qa_run: bool,
    read_from_datalake: bool,
    start_date: str,
    end_date: str,
) -> dict:
    """Process a client and channel and return the results.

    Args:
        data_report (list): List of the data report by client/channel
        client_key (str): The client key.
        channel (str): The channel.
        attr_name (str): The attr_name to read from Athena.
        is_local (bool): Flag indicating if the script is running locally.
        is_qa_run (bool): Flag indicating if the script is running in qa.
        read_from_datalake (bool): True if query from elasticity_datalake.sql.
        start_date (str): The start_date to read data.
        end_date (str): The end_date.

    Returns:
        dict: The results of processing.
    """
    error_counter = logging_error.ErrorCounter()
    logging.getLogger().addHandler(error_counter)
    try:
        logging.info(f"Processing {client_key} - {channel} - attr: {attr_name}")
        start_time = datetime.now()

        df_by_price, _, total_end_date_uid, df_revenue_uid, total_revenue = (
            preprocessing.read_and_preprocess(
                client_key=client_key,
                channel=channel,
                read_from_datalake=read_from_datalake,
                start_date=start_date,
                end_date=end_date,
            )
        )

        if df_by_price is None:
            raise ValueError("Error: df_by_price is None")

        if df_by_price.empty:
            raise ValueError("Error: df_by_price is Empty")

        s3io.write_dataframe_to_s3(
            file_name=f"df_by_price_{client_key}_{channel}_{end_date}.parquet",
            xdf=df_by_price,
            s3_dir="data_science/eval_results/elasticity/",
        )

        df_results = run_experiment_for_uids_parallel(
            df_by_price[~df_by_price["outlier_quantity"]],
            price_col="round_price",
            quantity_col="units",
            weights_col="days",
        )

        if attr_name:
            logging.info(f"Running group elasticity - attr: {attr_name}")

            # FOR METHOD WITH NO FILTER (KEEP IT FOR NOW)
            # df_by_price_GROUP = df_by_price[df_by_price["units"] > 0.001]
            # df_group = data_for_group_elasticity(
            #     df_by_price_GROUP, client_key, channel, attr_name
            # )

            df_group = data_for_group_elasticity(
                df_by_price=df_by_price,
                client_key=client_key,
                channel=channel,
                end_date=end_date,
                attr_name=attr_name,
            )

            df_results = add_group_elasticity(df_group, df_results)
        else:
            logging.info(f"Skipping group elasticity - attr: {attr_name}")
            df_results["result_to_push"] = df_results["quality_test"]
            df_results["type"] = "uid"

        df_results = df_results.merge(df_revenue_uid, on="uid", how="left")

        logging.info(
            f"Quality test: {df_results[df_results.result_to_push].quality_test.value_counts()}"
        )
        logging.info(
            f"Test high: {df_results[df_results.result_to_push].quality_test_high.value_counts()}"
        )
        logging.info(f"Type: {df_results[df_results.result_to_push]['type'].value_counts()}")

        upload_elasticity_data_to_athena(
            client_key=client_key,
            channel=channel,
            end_date=end_date,
            df_upload=df_results,
            table_name=app_state.models_monitoring_table_name,  # projects_kpis_table_name,
        )

        s3io.write_dataframe_to_s3(
            file_name=f"elasticity_{client_key}_{channel}_{end_date}.csv",
            xdf=df_results,
            s3_dir="data_science/eval_results/elasticity/",
        )

        plot_demands.run_save_graph_top10(df_results, df_by_price, client_key, channel, end_date)

        actions_list = generate_actions_list(df_results, client_key, channel)

        input_args = dc.WriteActionsListKWArgs(
            client_key=client_key,
            channel=channel,
            is_local=is_local,
            qa_run=is_qa_run,
        )

        _write_actions_list(
            actions_list=actions_list,
            input_args=input_args,
        )

        runtime_duration = (datetime.now() - start_time).total_seconds() / 60

        data_report = report.generate_run_report(
            client_key=client_key,
            channel=channel,
            total_uid=total_end_date_uid,
            results_df=df_results,
            runtime_duration=runtime_duration,
            total_revenue=total_revenue,
            error_count=error_counter.error_count,
            end_date=end_date,
        )

        upload_elasticity_data_to_athena(
            client_key=client_key,
            channel=channel,
            end_date=end_date,
            df_upload=data_report,
            table_name=app_state.projects_kpis_table_name,
        )

        write_graphs.save_distribution_graph(
            client_key=client_key,
            channel=channel,
            total_uid=total_end_date_uid,
            df_report=data_report,
            end_date=end_date,
            s3_dir=app_state.s3_eval_results_dir + "/graphs/",
        )

    except (KeyError, pd.errors.EmptyDataError, ValueError) as e:
        logging.error(f"Error processing {client_key} - {channel}: {e}")
        error_info = traceback.format_exc()
        logging.error(f"Error occurred in {__file__} - {e} \n{error_info}")

        # data_report = report.generate_error_report(
        #     client_key=client_key,
        #     channel=channel,
        #     error_count=error_counter.error_count,
        # )

        # upload_elasticity_data_to_athena(
        # client_key=client_key,
        # channel=channel,
        # end_date=end_date,
        # df_upload=data_report,
        # table_name=app_state.projects_kpis_table_name,
        # )

    return


def run() -> None:
    """Main function to run the elasticity job."""
    (client_keys_map, is_local, is_qa_run, start_date, end_date) = setup_environment()
    for client_key in client_keys_map:
        channels_list = client_keys_map[client_key]["channels"]
        attr_name = client_keys_map[client_key]["attr_name"]
        read_from_datalake = client_keys_map[client_key].get("read_from_datalake", False)
        for channel in channels_list:
            process_client_channel(
                client_key=client_key,
                channel=channel,
                attr_name=attr_name,
                is_local=is_local,
                is_qa_run=is_qa_run,
                read_from_datalake=read_from_datalake,
                start_date=start_date,
                end_date=end_date,
            )


if __name__ == "__main__":
    run()
