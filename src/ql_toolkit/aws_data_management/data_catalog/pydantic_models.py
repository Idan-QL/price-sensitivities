"""This module contains data classes that are used in the data lake module."""

from pydantic import BaseModel, Field


class AthenaUploadConfig(BaseModel):
    """Configuration model for Athena data upload."""

    client_key: str = Field(..., description="The client key for the Athena partition.")
    channel: str = Field(..., description="The channel associated with the Athena partition.")
    table_name: str = Field(..., description="The name of the Glue table to upload data to.")
    # nan_fill_value: int = Field(..., description="Value to replace NaN entries in the DataFrame.")
    # inf_fill_value: int = Field(
    #     ..., description="Value to replace inf/-inf entries in the DataFrame."
    # )


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
