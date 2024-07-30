"""Module for generating actions list from the results DataFrame."""

import logging
from datetime import datetime
from typing import List

import pandas as pd

from elasticity.utils.consts import CODE_VERSION
from ql_toolkit.attrs.action_list import create_actions_list


def generate_actions_list(df_results: pd.DataFrame, client_key: str, channel: str) -> List[dict]:
    """Generate actions list from the results DataFrame."""
    # Combine best_a, best_b, and best_model into a dictionary
    df_results["qlia_elasticity_model"] = (
        df_results[["best_a", "best_b", "best_model"]]
        .rename(columns={"best_a": "a", "best_b": "b", "best_model": "model"})
        .to_dict(orient="records")
    )

    # Add the code_version to each dictionary in the column
    df_results["qlia_elasticity_model"] = df_results["qlia_elasticity_model"].apply(
        lambda model: {**model, "code_version": CODE_VERSION}
    )

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
