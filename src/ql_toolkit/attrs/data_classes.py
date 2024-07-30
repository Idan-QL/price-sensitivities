"""This module contains data classes for the write module."""

from typing import Optional

from pydantic import BaseModel


class WriteAttributesKWArgs(BaseModel):
    """Data class for write_attributes keyword arguments.

    Attributes:
        client_key (str): The client name
        channel (str): The channel
        is_local (bool): Whether to write to local or s3
        qa_run (bool): Write results to a test directory
    """

    client_key: str
    channel: str
    is_local: bool
    qa_run: bool


class WriteActionsListKWArgs(BaseModel):
    """Data class for write_actions_list keyword arguments.

    Attributes:
        client_key (str): The retailer name
        channel (str): The channel
        qa_run (bool): Write results to a test directory
        filename_prefix (str): The prefix of the file name. Defaults to None
        is_local (bool): Whether to write to local or s3. Defaults to False.
        chunk_size (int): The size of each chunk. Defaults to 5000.
    """

    client_key: str
    channel: str
    qa_run: bool
    is_local: bool = False
    filename_prefix: Optional[str] = None
    chunk_size: int = 5000
