"""Pydantic models for the QL Toolkit API."""

import logging
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ResultItem(BaseModel):
    """Schema for individual result items in the results list.

    Attributes:
        entity_type (str): The type of entity.
        entity_id (str): The entity ID.
        attribute_name (str): The name of the attribute.
        value (Dict[str, Any]): The value associated with the attribute.
    """

    entity_type: str = Field(description="The type of entity.")
    entity_id: str = Field(description="The entity ID.")
    attribute_name: str = Field(description="The name of the attribute.")
    value: dict[str, Any] = Field(description="The value associated with the attribute.")


class WriteModelResults(BaseModel):
    """Schema for the request body when writing new model results.

    Attributes:
        client_key (str): The client key.
        channel (str): The channel.
        model (str): The model used.
        validity_days (int): The number of days the results are valid for. It must be at least 1.
        results (List[ResultItem]): A list of result items. The list cannot be empty.
        chunk_size (int): The size of each chunk of the results list (default 1000). The chunk size
            must be greater than 0 and less than or equal to 2500.
    """

    client_key: str = Field(description="The client key.")
    channel: str = Field(description="The channel.")
    model: str = Field(description="The model used.")
    validity_days: int = Field(
        description="The number of days the results are valid for."
    )
    results: list[ResultItem] = Field(description="A list of result items.")
    chunk_size: int = Field(description="The size of each chunk.", default=1000)

    @field_validator("validity_days")
    @classmethod
    def validate_validity_days(cls, value: int) -> int:
        """Validate the 'validity_days' field to ensure it is greater than 0."""
        if value <= 1:
            err_msg = "'validity_days' must be greater than 1"
            logging.error(err_msg)
            raise ValueError(err_msg)

        return value

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        """Validate the 'chunk_size' field to ensure it is greater than 0."""
        if value <= 0:
            err_msg = "'chunk_size' must be greater than 0"
            logging.error(err_msg)
            raise ValueError(err_msg)

        max_chunk_size = 2500
        if value > max_chunk_size:
            err_msg = f"'chunk_size' cannot exceed {max_chunk_size}"
            logging.error(err_msg)
            raise ValueError(err_msg)

        return value

    @field_validator("results")
    @classmethod
    def validate_results(cls, value: list[ResultItem]) -> list[ResultItem]:
        """Validate the 'results' field to ensure it is not empty."""
        if not value:
            err_msg = "'results' list cannot be empty"
            logging.error(err_msg)
            raise ValueError(err_msg)

        return value
