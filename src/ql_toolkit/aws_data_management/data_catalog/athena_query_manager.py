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

    def convert_dtypes(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Convert numerical data types to optimize memory usage while preserving null values.

        This function converts:
        - Nullable integer columns (`Int64`) to 32-bit nullable integers (`Int32`).
        - Floating-point columns (`float64`) to 32-bit floating-point numbers (`float32`).

        Args:
            data_df (pd.DataFrame): The input DataFrame containing numerical and other data.

        Returns:
            pd.DataFrame: The DataFrame with optimized numerical data types.
        """
        skipped_columns = []

        # Process nullable integer columns
        data_df, skipped_int_cols = self._convert_columns(
            data_df=data_df,
            dtype_filter="Int64",
            target_dtype="Int32",
            dtype_name="integer",
        )
        skipped_columns.extend(skipped_int_cols)

        # Process floating-point columns
        data_df, skipped_float_cols = self._convert_columns(
            data_df=data_df,
            dtype_filter="float64",
            target_dtype="float32",
            dtype_name="float",
        )
        skipped_columns.extend(skipped_float_cols)

        # Logging summary
        logging.info(
            f"DataFrame data types converted! "
            f"{len(data_df.select_dtypes(include=['Int32']))} integer columns "
            f"and {len(data_df.select_dtypes(include=['float32']))} float columns updated."
        )
        if skipped_columns:
            logging.warning(
                f"Skipped the following columns due to conversion issues: {skipped_columns}"
            )

        return data_df

    @staticmethod
    def _convert_columns(
        data_df: pd.DataFrame, dtype_filter: str, target_dtype: str, dtype_name: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """Helper function to convert specific column types in a DataFrame.

        Args:
            data_df (pd.DataFrame): The input DataFrame.
            dtype_filter (str): The dtype to filter columns by (e.g., "Int64").
            target_dtype (str): The target dtype for conversion (e.g., "Int32").
            dtype_name (str): Name of the type being converted for logging purposes.

        Returns:
            tuple[pd.DataFrame, list[str]]: The updated DataFrame and a list of skipped columns.
        """
        skipped_columns = []
        cols_to_convert = data_df.select_dtypes(include=[dtype_filter]).columns

        for col in cols_to_convert:
            if not pd.api.types.is_numeric_dtype(data_df[col]):
                skipped_columns.append(col)
                continue
            try:
                data_df[col] = data_df[col].astype(target_dtype)
            except ValueError as err:
                logging.warning(
                    f"Skipping conversion of column '{col}' to {dtype_name}. Error: {err}"
                )
                skipped_columns.append(col)

        return data_df, skipped_columns

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
