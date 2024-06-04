"""This module contains functions to write actions to a file."""

# import json
from datetime import datetime

import numpy as np

from ql_toolkit.config.runtime_config import app_state


def create_actions_list(
    res_list: list, client_key: str, channel: str, attr_names: list[str]
) -> list:
    """Create a generic actions list.

    The date of the calculation is set to the current date and added to the action,
    where the attribute is
    internal and its name is set to f"qlia_{app_state.project_name}_calc_date".

    Args:
        res_list (list): List of tuples containing the data.
        client_key (str): Client key.
        channel (str): Channel.
        attr_names (list[str]): List of attribute names in the order they appear in res_list tuples.

    Returns:
        list: List of actions.

    Example:
        attr_names = [
        "uid",
        "fcst_val",
        "fcst_reason",
        "upper_ci",
        "lower_ci",
        "max_units_sold",
        "max_shelf_price",
        "fcst_shelf_price",
        "list_of_ints",
        "arr_of_floats"
        ]
        res_list = [
            ("uid1", 100, "reason1", 110.0, 90.0, 200, 10.0, 9.0, [1, 2, 3],
            np.array([1.4, 2.5, 3.7])),
            ("uid2", 150, "reason2", 160.0, 140.0, 250, 15.0, 14.0, [4, 5, 6],
            np.array([4.5, 5.6, 6.7]))
        ]  # Your list of tuples
        actions_list = create_actions_list(
            res_list=res_list, client_key="client_key", channel="channel", attr_names=attr_names
        )
        print(actions_list)
    """
    actions_list = []
    calc_date_str = datetime.now().strftime(app_state.date_format)

    for res_tup in res_list:
        if not res_tup:
            continue

        # Create a dictionary from attribute names and values in res_tup
        attr_values = dict(zip(attr_names, res_tup))
        attr_values[f"qlia_{app_state.project_name}_calc_date"] = calc_date_str

        # Create action using the generated dictionary
        action = create_action(
            uid=attr_values.get("uid"),
            client_key=client_key,
            channel=channel,
            attr_values=attr_values,
        )

        actions_list.append(action)

    return actions_list


def create_action(
    uid: str,
    client_key: str,
    channel: str,
    attr_values: dict,
) -> dict:
    """Create a generic action dictionary.

    Args:
        uid (str): Unique identifier.
        client_key (str): Client key.
        channel (str): Channel.
        attr_values (dict): Dictionary of other attributes and their values.

    Returns:
        dict: Action dictionary.
    """
    action = {
        "action": "product_update",
        "changes": [
            {
                "uid": uid,
                "client_key": client_key,
                "channel": channel,
                "attrs": [],
            }
        ],
    }

    for attr_name, value in attr_values.items():
        if attr_name not in ["uid"]:  # Exclude already handled attributes
            if isinstance(value, (list, np.ndarray)):
                value = list_to_delimited_string(value)
            # val = json.dumps(value)
            # action["changes"][0]["attrs"].append({"name": attr_name, "value": val})
            action["changes"][0]["attrs"].append({"name": attr_name, "value": value})

    return action


def list_to_delimited_string(
    input_list: list[int] | list[float], delimiter: str = "|"
) -> str:
    """Converts a list into a delimited string.

    Args:
        input_list (list): The input list to be converted into a delimited string.
        delimiter (str, optional): The delimiter to separate the elements in the resulting string.
            Defaults to '|'.

    Returns:
        str: The delimited string.

    """
    # Convert all the elements in the list to string
    str_list = map(str, input_list)

    # Join the elements with the specified delimiter
    return delimiter.join(str_list)
