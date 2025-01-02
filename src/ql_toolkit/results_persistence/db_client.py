"""Database client module for writing payloads to the database."""

import asyncio
import logging
from typing import Optional
from urllib.parse import urljoin

import httpx
from tenacity import retry, stop_after_attempt, wait_fixed

from ql_toolkit.results_persistence import db_payload
from ql_toolkit.results_persistence.api_utils import get_ds_service_api_url
from ql_toolkit.results_persistence.pydantic_models import WriteModelResults


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def send_post_request(
    client: httpx.AsyncClient, url: str, payload: dict, semaphore: asyncio.Semaphore
) -> None:
    """Send a POST request asynchronously with concurrency control and retry logic."""
    async with semaphore:
        try:
            response = await client.post(url=url, json=payload)
            response.raise_for_status()
            logging.info(f"Success: {response.status_code} - {response.json()}")
        except httpx.HTTPStatusError as exc:
            logging.error(f"HTTP error occurred: {exc.response.status_code} - {exc.response.text}")
            raise
        except Exception as exc:
            logging.error(f"An error occurred while sending post request: {exc}")
            raise


async def write_payloads_to_db(
    api_url: str, payloads: list[dict], concurrency_limit: int = 4
) -> None:
    """Write results to the database asynchronously."""
    logging.info(f"Writing {len(payloads)} payloads to the database...")
    semaphore = asyncio.Semaphore(concurrency_limit)
    failed_payloads = []

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            tasks = [
                send_post_request(client=client, url=api_url, payload=payload, semaphore=semaphore)
                for payload in payloads
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for payload, result in zip(payloads, results):
            if isinstance(result, Exception):
                failed_payloads.append((payload, result))

    except Exception as exc:
        err_msg = f"An error occurred while writing payloads: {exc}"
        logging.error(err_msg)
        raise exc

    if failed_payloads:
        err_msg = (
            f"Failed to write all payloads to the database: "
            f"({len(failed_payloads)} payloads not written)"
        )
        logging.error(err_msg)
        raise Exception(err_msg)


# After sending all the data chunks
async def finalize_results_cycle(api_url: str, payload: dict) -> None:
    """Finalize the results cycle by updating metadata status.

    This function sends a POST request to the database API endpoint to finalize the results cycle.
    It updates the metadata status for the given `client key`, `channel`, `model`, and
    `computed_at` combination to 'ACTIVE'. Following that, the metadata status of older records
    is updated to 'DEPRECATED'.

    Args:
        api_url (str): The URL of the database API endpoint.
        payload (dict): The request body containing the following fields:
            - client_key (str): The client key.
            - channel (str): The channel.
            - model (str): The model used.
            - computed_at_str (str): The timestamp when the results were computed.
            - status (str): The status to update the metadata to.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(url=api_url, json=payload)
            response.raise_for_status()
            logging.info(f"Finalization successful: {response.json()}")
        except httpx.HTTPStatusError as exc:
            logging.error(
                f"HTTP error during finalization: {exc.response.status_code} - "
                f"{exc.response.text}"
            )
            raise
        except Exception as exc:
            logging.error(f"An error occurred during finalization: {exc}")
            raise


async def write_model_results_to_db(
    write_model_results_struct: WriteModelResults,
    api_base_url: Optional[str] = None,
) -> None:
    """Write model results to the database.

    This function generates payloads from the model results and writes them to the database API
    endpoint. It then finalizes the results cycle by updating the metadata status.

    Args:
        write_model_results_struct (WriteModelResults): The request body containing model results
            data.
        api_base_url (Optional[str]): The base URL of the ds-service API endpoint.
    """
    payloads_list, computed_at_str = db_payload.generate_payloads(
        write_model_results_struct=write_model_results_struct
    )

    if api_base_url is None:
        api_base_url = get_ds_service_api_url()

    write_results_url = urljoin(base=api_base_url, url="write-results")

    try:
        await write_payloads_to_db(api_url=write_results_url, payloads=payloads_list)
        # If all payloads are successfully written, finalize with 'ACTIVE' status
        final_status = "ACTIVE"
    except Exception as exc:
        err_msg = f"An error occurred while writing results: {exc}"
        logging.error(err_msg)
        # If an error occurs, finalize with 'FAILED' status
        final_status = "FAILED"

    finalize_results_url = urljoin(base=api_base_url, url="finalize-write-results-cycle")
    payload = {
        "client_key": write_model_results_struct.client_key,
        "channel": write_model_results_struct.channel,
        "model": write_model_results_struct.model,
        "computed_at": computed_at_str,
        "final_status": final_status,
    }
    await finalize_results_cycle(
        api_url=finalize_results_url,
        payload=payload,
    )
