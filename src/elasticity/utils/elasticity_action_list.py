"""Module for generating actions list from the results DataFrame."""

import logging
from datetime import datetime
from typing import List

import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from ql_toolkit.results_persistence.old_action_list import data_classes as dc
from ql_toolkit.results_persistence.old_action_list.action_list import (
    create_actions_list,
)
from ql_toolkit.results_persistence.old_action_list.write import _write_actions_list


def process_actions_list(
    df_results: pd.DataFrame,
    data_fetch_params: DataFetchParameters,
    is_qa_run: bool,
) -> None:
    """Generate and write the actions list based on the results.

    Args:
        df_results (pd.DataFrame): The DataFrame containing experiment results.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching
        such as client key and channel.
        is_qa_run (bool): Flag indicating if the script is running in QA environment.

    Returns:
        None
    """
    actions_list = generate_actions_list(
        df_results=df_results,
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
    )

    input_args = dc.WriteActionsListKWArgs(
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
        qa_run=is_qa_run,
    )

    _write_actions_list(
        actions_list=actions_list,
        input_args=input_args,
    )


def generate_actions_list(df_results: pd.DataFrame, client_key: str, channel: str) -> List[dict]:
    """Generate actions list from the results DataFrame."""
    attr_cs = [
        "uid",
        "qlia_elasticity_model",
        "details",
    ]
    df_actions = df_results[df_results.result_to_push][attr_cs]
    df_actions["qlia_elasticity_calc_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    attr_names = [
        "uid",
        "qlia_elasticity_model",
        "qlia_elasticity_details",
        "qlia_elasticity_calc_date",
    ]
    res_list = [tuple(row) for row in df_actions.to_numpy()]
    logging.info(f"len of res_list: {len(res_list)}")

    return create_actions_list(
        res_list=res_list, client_key=client_key, channel=channel, attr_names=attr_names
    )


def generate_delete_actions_list(
    uid_to_delete_list: List[str],
    attributes_list: List[str],
    client_key: str,
    channel: str,
) -> List[dict]:
    """Generate an actions list for deleting attributes from a list of UIDs.

    This function creates a list of actions for deleting specified attributes for each UID
    in the provided list. Each action is represented as a tuple where the attributes to be
    deleted are marked with the string 'delete!'.

    Args:
        uid_to_delete_list (List[str]): A list of unique identifiers (UIDs) for which attributes
        need to be deleted.
        attributes_list (List[str]): A list of attribute names that need to be deleted for each UID.
        client_key (str): A unique key for the client making the request.
        channel (str): The channel through which the request is made.

    List:
        List[dict]: A List containing the actions dictionary ready to be processed.
    """
    df_actions = pd.DataFrame({"uid": uid_to_delete_list})
    for c in attributes_list:
        df_actions[c] = "delete!"

    res_list = [tuple(row) for row in df_actions.to_numpy()]
    logging.info(f"len of res_list: {len(res_list)}")

    return create_actions_list(
        res_list=res_list,
        client_key=client_key,
        channel=channel,
        attr_names=["uid", *attributes_list],
        delete_actions=True,
    )
