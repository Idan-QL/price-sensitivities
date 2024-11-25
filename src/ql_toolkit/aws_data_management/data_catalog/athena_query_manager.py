"""This module defines the functionality to build and execute a Trino query on AWS Athena."""

import logging
from os import path
from typing import Any

import pandas as pd
from pyathena import connect
from pyathena.error import DatabaseError, OperationalError
from pyathena.pandas.cursor import PandasCursor

from ql_toolkit.application_state.manager import app_state


class AthenaQuery:
    """This class defines the functionality to build and execute a Trino query on AWS Athena.

    It is designed to load a query from a SQL file, format it with the necessary arguments
    and execute it on AWS Athena.

    Attributes:
        client_key (str): The client key.
        channel (str): The channel.
        kwargs (dict): Additional keyword arguments, necessary for the formatting of the query.
    """

    def __init__(
        self, client_key: str, channel: str, file_name: str, **kwargs: dict[str, Any]
    ) -> None:
        """Initialize the AthenaQuery class.

        Args:
            client_key (str): The client key.
            channel (str): The channel.
            file_name (str): The name of the SQL file containing the query, without the
                `.sql` postfix.
            **kwargs: Additional keyword arguments.
        """
        self.client_key = client_key
        self.channel = channel
        self.file_name = f"{file_name}.sql"
        self.kwargs = kwargs

    def load_query_from_file(self) -> str:
        """Load the query string to be formatted from a SQL file.

        Returns:
            str: The raw query string.

        Raises:
            FileNotFoundError: If the SQL file does not exist in the specified directory.
        """
        queries_dir = path.join(path.dirname(__file__), "query_scripts")
        file_path = path.join(queries_dir, self.file_name)
        logging.info(f"Loading query from file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError as err:
            raise FileNotFoundError(
                f"Query file {self.file_name} not found in {queries_dir}"
            ) from err

    def build_query(self) -> str:
        """Build the query string by formatting the raw query with the provided parameters.

        Returns:
            str: The formatted query string.
        """
        raw_query = self.load_query_from_file()
        logging.info(
            f"Building query with client_key={self.client_key}, "
            f"channel={self.channel}, kwargs={self.kwargs}"
        )
        return raw_query.format(client_key=self.client_key, channel=self.channel, **self.kwargs)

    @staticmethod
    def convert_dtypes(data_df: pd.DataFrame) -> pd.DataFrame:
        """Convert Int64 columns to int32 and float64 columns to float32 in the DataFrame.

        Args:
            data_df (pd.DataFrame): The DataFrame to convert.

        Returns:
            pd.DataFrame: The DataFrame with converted data types.
        """
        # Convert int64 to int32
        int_cols = data_df.select_dtypes(include=["Int64"]).columns
        data_df[int_cols] = data_df[int_cols].astype("int32")

        # Convert float64 to float32
        float_cols = data_df.select_dtypes(include=["float64"]).columns
        data_df[float_cols] = data_df[float_cols].astype("float32")

        logging.info("Data types converted!")
        return data_df

    def execute_query(self) -> pd.DataFrame:
        """Execute the built query on AWS Athena and return the results as a pandas DataFrame.

        Returns:
            pd.DataFrame: The result of the executed query as a pandas DataFrame.

        Raises:
            DatabaseError: If there is an issue with the database during query execution.
            OperationalError: If there is an operational error during query execution.
            Exception: If any other exception occurs during query execution.
        """
        query = self.build_query()
        try:
            logging.info("Executing query...")
            with connect(
                s3_staging_dir=app_state.s3_athena_staging_dir,
                region_name=app_state.aws_region,
                cursor_class=PandasCursor,
            ) as conn:
                athena_df = conn.cursor().execute(query).as_pandas()
            logging.info("Query executed successfully.")
        except (DatabaseError, OperationalError, Exception) as err:
            logging.error(f"Error executing query: {err}")
            return pd.DataFrame()

        return self.convert_dtypes(athena_df)
