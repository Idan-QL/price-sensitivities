"""This module contains functions to write attributes to a file."""

import asyncio
import logging

import pandas as pd

from elasticity.data.configurator import DataFetchParameters
from elasticity.model import run_model
from ql_toolkit.application_state.manager import app_state
from ql_toolkit.results_persistence import db_client, pydantic_models


def upload_elasticity_results_to_attr_db(
    df_results: pd.DataFrame,
    data_fetch_params: DataFetchParameters,
) -> None:
    """Upload elasticity results to the attribute database.

    Args:
        df_results (pd.DataFrame): The DataFrame containing experiment results.
        data_fetch_params (DataFetchParameters): Parameters related to data fetching.

    Returns:
        None

    """
    attr_cs = [
        "uid",
        "qlia_elasticity_model",
        "type",
        "quality_test",
        "quality_test_high",
    ]
    df_actions = df_results[df_results.result_to_push][attr_cs].copy()

    df_actions["qlia_elasticity_quality"] = df_actions.apply(
        lambda row: run_model.make_details(row["quality_test"], row["quality_test_high"]),
        axis=1,
    )

    df_actions = df_actions.drop(columns=["quality_test", "quality_test_high"])
    attr_names = [
        "uid",
        "model",
        "explanation",
        "fit_quality",
    ]
    df_actions.columns = attr_names
    df_flattened = pd.concat(
        [df_actions.drop(columns=["model"]), df_actions["model"].apply(pd.Series)], axis=1
    )
    df_flattened = df_flattened.rename(columns={
        "model": "demand_model",
        "code_version": "version",
    })
    write_model_results(
        results_df=df_flattened,
        client_key=data_fetch_params.client_key,
        channel=data_fetch_params.channel,
    )


def write_model_results(
    results_df: pd.DataFrame,
    client_key: str,
    channel: str,
) -> None:
    """Write model results to attr db.

    Args:
        results_df (pd.DataFrame): DataFrame containing the results
        client_key (str): Client key
        channel (str): Channel

    Returns:
        None
    """
    db_struct = pydantic_models.WriteModelResults(
        client_key=client_key,
        channel=channel,
        model=app_state.project_name,
        validity_days=32,
        results=[
            _create_result_item(
                result=row,
            )
            for _, row in results_df.iterrows()  # Iterate over rows in the DataFrame
        ],
    )
    asyncio.run(
        db_client.write_model_results_to_db(
            write_model_results_struct=db_struct,
        )
    )
    logging.info(
        f"{len(db_struct.results)} \
            elasticity results were sent to attr DB"
    )


def _create_result_item(result: pd.Series) -> pydantic_models.ResultItem:
    """Create a ResultItem from a series."""
    return pydantic_models.ResultItem(
        entity_type="UID",
        entity_id=result["uid"],
        attribute_name=app_state.project_name,
        value=result.drop("uid").to_dict(),
    )
