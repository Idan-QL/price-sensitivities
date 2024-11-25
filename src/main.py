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
from elasticity.data.configurator import DataColumns, DataFetchParameters, DateRange
from elasticity.data.preprocessing import run_preprocessing, save_preprocess_to_s3

# from elasticity.data.group import data_for_group_elasticity
from elasticity.data.utils import initialize_dates
from elasticity.model.group import handle_group_elasticity
from elasticity.model.run_model import run_experiment_for_uids_parallel
from elasticity.utils import cli_default_args
from elasticity.utils.elasticity_action_list import process_actions_list
from elasticity.utils.utils import log_environment_mode
from elasticity.utils.write import upload_elasticity_data_to_athena

# from ql_toolkit.attrs import data_classes as dc
# from ql_toolkit.attrs.write import _write_actions_list
from ql_toolkit.application_state.manager import app_state
from ql_toolkit.env_setup.initialize_runtime import run_setup

# from ql_toolkit.s3 import io_tools as s3io
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
    config_dict, client_keys_map = run_setup(
        cli_args_dict=cli_default_args.args_kv,
        google_sheet_keep_cols=["client_keys", "channels", "attr_name"],
    )
    logging.info("config_dict: %s", config_dict)
    logging.info("client_keys_map: %s", client_keys_map)

    date_range = initialize_dates()

    is_local = config_dict.get("local", False)
    is_qa_run = app_state.results_type == "qa"
    log_environment_mode(is_local=is_local, is_qa_run=is_qa_run)

    return (client_keys_map, is_local, is_qa_run, date_range)


def log_quality_tests(df_results: pd.DataFrame) -> None:
    """Log the results of the quality tests.

    Args:
        df_results (pd.DataFrame): The DataFrame containing experiment results.

    Returns:
        None
    """
    logging.info(
        f"Quality test: {df_results[df_results.result_to_push].quality_test.value_counts()}"
    )
    logging.info(
        f"Test high: {df_results[df_results.result_to_push].quality_test_high.value_counts()}"
    )
    logging.info(f"Type: {df_results[df_results.result_to_push]['type'].value_counts()}")


def process_client_channel(
    data_fetch_params: DataFetchParameters,
    date_range: DateRange,
    is_qa_run: bool,
) -> None:
    """Process a client and channel.

    Args:
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.
        date_range (DateRange): The date range for fetching the data.
        is_local (bool): Flag indicating if the script is running locally.
        is_qa_run (bool): Flag indicating if the script is running in qa.

    Returns:
        dict: The results of processing.
    """
    error_counter = logging_error.ErrorCounter()
    logging.getLogger().addHandler(error_counter)

    data_columns = DataColumns()
    logging.info(
        f"Processing {data_fetch_params.client_key} - {data_fetch_params.channel}"
        f"- attr: {data_fetch_params.attr_name}"
    )

    start_time = datetime.now()
    try:
        # Step 1: Preprocessing and data loading
        df_by_price, df_revenue_uid, total_end_date_uid, total_revenue = run_preprocessing(
            data_fetch_params=data_fetch_params, date_range=date_range, data_columns=data_columns
        )

        # Step 2: Save the data to S3
        save_preprocess_to_s3(
            df_by_price=df_by_price,
            data_fetch_params=data_fetch_params,
            date_range=date_range,
            is_qa_run=is_qa_run,
        )

        # Step 3: Run the experiment
        df_results = run_experiment_for_uids_parallel(
            df_input=df_by_price[~df_by_price["outlier_quantity"]], data_columns=data_columns
        )

        # Step 4: Handle group elasticity if needed
        df_results = handle_group_elasticity(
            df_by_price=df_by_price,
            data_fetch_params=data_fetch_params,
            date_range=date_range,
            df_results=df_results,
            data_columns=data_columns,
        )

        # Step 5: Merge with revenue data
        df_results = df_results.merge(df_revenue_uid, on="uid", how="left")
        log_quality_tests(df_results=df_results)

        # Step 6: Action list and upload to athena
        upload_elasticity_data_to_athena(
            data_fetch_params=data_fetch_params,
            end_date=date_range.end_date,
            df_upload=df_results,
            table_name=app_state.models_monitoring_table_name,
        )
        process_actions_list(
            df_results=df_results,
            data_fetch_params=data_fetch_params,
            is_qa_run=is_qa_run,
        )

        # Step 7: Save graphs
        plot_demands.run_save_graph_top10(
            df_results=df_results,
            df_by_price=df_by_price,
            data_fetch_params=data_fetch_params,
            end_date=date_range.end_date,
        )

        # Step 8: Build and save report to Athena
        runtime_duration = (datetime.now() - start_time).total_seconds() / 60

        data_report = report.generate_run_report(
            data_fetch_params=data_fetch_params,
            total_uid=total_end_date_uid,
            results_df=df_results,
            runtime_duration=runtime_duration,
            total_revenue=total_revenue,
            error_count=error_counter.error_count,
            end_date=date_range.end_date,
            is_qa_run=is_qa_run,
        )

        write_graphs.save_distribution_graph(
            data_fetch_params=data_fetch_params,
            total_uid=total_end_date_uid,
            df_report=data_report,
            end_date=date_range.end_date,
            s3_dir=app_state.s3_eval_results_dir + "/graphs/",
        )

        upload_elasticity_data_to_athena(
            data_fetch_params=data_fetch_params,
            end_date=date_range.end_date,
            df_upload=data_report,
            table_name=app_state.projects_kpis_table_name,
        )

    except (KeyError, pd.errors.EmptyDataError, ValueError) as e:
        logging.error(
            f"Error processing {data_fetch_params.client_key} - {data_fetch_params.channel}: {e}"
        )
        error_info = traceback.format_exc()
        logging.error(f"Error occurred in {__file__} - {e} \n{error_info}")

    return


def run() -> None:
    """Main function to run the elasticity job."""
    (client_keys_map, _, is_qa_run, date_range) = setup_environment()
    for client_key in client_keys_map:

        channels_list = client_keys_map[client_key]["channels"]
        attr_name = client_keys_map[client_key]["attr_name"]
        read_from_datalake = client_keys_map[client_key].get("read_from_datalake", False)

        for channel in channels_list:
            data_fetch_params = DataFetchParameters(
                client_key=client_key,
                channel=channel,
                attr_name=attr_name,
                read_from_datalake=read_from_datalake,
            )

            process_client_channel(
                data_fetch_params=data_fetch_params,
                date_range=date_range,
                is_qa_run=is_qa_run,
            )


if __name__ == "__main__":
    run()
