"""This module contains data classes that are used in the data lake module."""

from pydantic import BaseModel


class GlueDBKeys(BaseModel):
    """This dataclass defines the keys for the Glue database, table and partitions.

    Attributes:
        database_name (str): The database name.
        table_name (str): The table name.
        client_key (str): The client key.
        channel (str): The channel.
        date (str): The date in string format.
    """

    database_name: str
    table_name: str
    client_key: str
    channel: str
    date: str
