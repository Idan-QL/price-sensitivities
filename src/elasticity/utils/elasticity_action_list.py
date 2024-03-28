"""Module for generating actions list from the results DataFrame."""
import logging
from datetime import datetime
from typing import Any, List, Tuple

import pandas as pd
from ql_toolkit.attrs.action_list import create_actions_list


def generate_actions_list(df_results: pd.DataFrame,
                          client_key: str,
                          channel: str) -> List[Tuple[Any, ...]]:
    """Generate actions list from the results DataFrame."""
    attr_cs = [
        "uid",
        "best_model_a",
        "best_model_b",
        "best_model",
        "power_elasticity",
    ]
    df_actions = df_results[df_results.quality_test][attr_cs]
    df_actions['qlia_elasticity_calc_date'] = datetime.now().strftime("%Y-%m-%d %H:%M")
    attr_names = [
        "uid",
        "qlia_elasticity_param1",
        "qlia_elasticity_param2",
        "qlia_elasticity_demand_model",
        "qlia_product_elasticity",
        "qlia_elasticity_calc_date",
    ]
    res_list = [tuple(row) for row in df_actions.to_numpy()]
    logging.info("len of res_list: %s", len(res_list))

    return create_actions_list(
        res_list=res_list,
        client_key=client_key,
        channel=channel,
        attr_names=attr_names)
