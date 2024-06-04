"""Module for generating actions list from the results DataFrame."""

import logging
from datetime import datetime
from typing import Any, List, Tuple

import pandas as pd

from elasticity.utils.consts import MIN_ELASTICITY
from ql_toolkit.attrs.action_list import create_actions_list


def generate_actions_list(
    df_results: pd.DataFrame,
    client_key: str,
    channel: str,
    min_elasticity: float = MIN_ELASTICITY,
) -> List[Tuple[Any, ...]]:
    """Generate actions list from the results DataFrame."""
    df_results["cap_elasticity"] = df_results["best_elasticity"].apply(
        lambda x: max(x, min_elasticity)
    )

    # Combine best_a, best_b, and best_model into a dictionary
    df_results["qlia_elasticity_model"] = (
        df_results[["best_a", "best_b", "best_model"]]
        .rename(columns={"best_a": "a", "best_b": "b", "best_model": "model"})
        .to_dict(orient="records")
    )

    attr_cs = [
        "uid",
        "cap_elasticity",
        "elasticity_level",
        "qlia_elasticity_model",
        "details",
    ]
    df_actions = df_results[df_results.result_to_push][attr_cs]
    df_actions["qlia_elasticity_calc_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    attr_names = [
        "uid",
        "elasticity_value",
        "elasticity_level",
        "qlia_elasticity_model",
        "qlia_elasticity_details",
        "qlia_elasticity_calc_date",
    ]
    res_list = [tuple(row) for row in df_actions.to_numpy()]
    logging.info(f"len of res_list: {len(res_list)}")

    return create_actions_list(
        res_list=res_list, client_key=client_key, channel=channel, attr_names=attr_names
    )
