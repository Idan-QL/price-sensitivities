"""This module contains functions to generate payloads to send to the database API endpoint."""

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

from ql_toolkit.results_persistence.pydantic_models import WriteModelResults


def split_list(data: list[Any], chunk_size: int) -> list[list[Any]]:
    """Splits a list into smaller chunks of a specified size."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


def generate_payloads(write_model_results_struct: WriteModelResults) -> tuple[list[dict], str]:
    """Generate a list of payloads to send to the database API endpoint.

    This function a list of model results and splits it into chunks to avoid large payloads.
    The computed_at and valid_until timestamps are set based on the current time.
    The payloads are formatted as dictionaries.

    Args:
        write_model_results_struct (WriteModelResults): The request body containing model
            results data.

    Returns:
        list[dict]: A list of payloads to send to the database API endpoint.
    """
    client_key = write_model_results_struct.client_key
    channel = write_model_results_struct.channel
    model = write_model_results_struct.model
    validity_days = write_model_results_struct.validity_days
    model_results_list = write_model_results_struct.results
    chunk_size = write_model_results_struct.chunk_size

    computed_at = datetime.now(tz=UTC)
    valid_until = computed_at + timedelta(days=validity_days)

    computed_at_str = computed_at.isoformat()
    valid_until_str = valid_until.isoformat()

    payloads = []
    for results_chunk in list(split_list(data=model_results_list, chunk_size=chunk_size)):
        # Convert each ResultItem to a dictionary
        results_list = [result_item.dict() for result_item in results_chunk]

        payload = {
            "client_key": client_key,
            "channel": channel,
            "model": model,
            "computed_at": computed_at_str,
            "valid_until": valid_until_str,
            "results": results_list,
        }
        payloads.append(payload)

    return payloads, computed_at_str


def get_string_hash(s: str) -> str:
    """Generate a hash string from a given string.

    Args:
        s (str): The input string to hash.

    Returns:
        str: The hash string.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(s.encode("utf-8"))
    hash_bytes = hash_obj.digest()

    return base64.urlsafe_b64encode(hash_bytes).decode("utf-8").rstrip("=")


def get_cid(channel: str, uid: str) -> str:
    """Generate a client ID from a channel and UID.

    Args:
        channel (str): The channel.
        uid (str): The UID.

    Returns:
        str: The client ID.
    """
    s = f"{channel}{uid}"
    return get_string_hash(s)
