"""This module contains the AthenaDataManager class that manages the Athena - Glue operations."""

import logging
from os import path
from typing import Dict, List, TypeAlias

import boto3
import pandas as pd
import pyarrow as pa

from ql_toolkit.arrow import io_tools as arrow_io
from ql_toolkit.arrow import utils as arrow_utils
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.data_lake.data_classes import GlueDBKeys
from ql_toolkit.s3 import io_tools as s3io

# Type alias for boto3 Glue client
GlueClient: TypeAlias = boto3.client


class AthenaDataManager:
    """Class that manages the Athena - Glue operations."""

    def __init__(
        self,
        region_name: str,
        database_s3_uri: str,
        glue_db_keys_dc: GlueDBKeys,
    ) -> None:
        """Initialize the AthenaDataManager.

        Args:
            region_name (str): The AWS region name.
            database_s3_uri (str): The S3 URI of the database.
            glue_db_keys_dc (GlueDBKeys): The data required to define the Glue DB, Table
                and Partitions.
        """
        self.region_name = region_name
        self.database_name = glue_db_keys_dc.database_name
        self.table_name = glue_db_keys_dc.table_name
        self.table_uri = str(path.join(database_s3_uri, glue_db_keys_dc.table_name))

        self.partition_values_list = self.get_partition_values_list(
            partition_keys_dc=glue_db_keys_dc
        )
        partition_partial_path = self.get_partition_partial_path(
            partition_values_list=self.partition_values_list
        )
        bucket_idx = self.table_uri.find(app_state.bucket_name) + len(app_state.bucket_name) + 1
        table_dir_prefix = self.table_uri[bucket_idx:]
        self.partition_path = path.join(table_dir_prefix, partition_partial_path)
        self.partition_uri = str(path.join(self.table_uri, partition_partial_path))

        self.file_name = (
            f"{glue_db_keys_dc.client_key}_{glue_db_keys_dc.channel}"
            f"_{glue_db_keys_dc.date}.parquet"
        )

        self.glue_client = self.get_glue_client(region_name=region_name)

        self.serialization_library = "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"

    def get_partition_values_list(self, partition_keys_dc: GlueDBKeys) -> List[str]:
        """Get the partition values list for the Athena table based on the table name.

        Args:
            partition_keys_dc (GlueDBKeys): The data required to define partitions.

        Returns:
            List[str]: The partition values list for the Athena table.

        Raises:
            ValueError: If the table name is not recognized.
        """
        valid_table_names = {
            app_state.models_monitoring_table_name,
            app_state.projects_kpis_table_name,
        }

        if self.table_name in valid_table_names:
            return [
                app_state.project_name,
                partition_keys_dc.client_key,
                partition_keys_dc.channel,
                partition_keys_dc.date,
            ]

        raise ValueError(f"Table name '{self.table_name}' not recognized.")

    @staticmethod
    def get_partition_partial_path(partition_values_list: List[str]) -> str:
        """Get the partition partial path for the Athena 'models_monitoring' table.

        The partial path is the path that is appended to the table URI to create the partition URI.
        For example, if the partition values are ['value1', 'value2', 'value3', 'value4'],
        the partial path will be 'project_name=value1/client_key=value2/channel=value3/date=value4'.

        Args:
            partition_values_list (List[str]): The partition values list.

        Returns:
            str: The partition partial path for the Athena table.
        """
        partition_keys = ["project_name", "client_key", "channel", "date"]
        return "/".join(
            [f"{key}={value}" for key, value in zip(partition_keys, partition_values_list)]
        )

    @staticmethod
    def get_glue_client(region_name: str) -> GlueClient:
        """Get the Glue client.

        Args:
            region_name (str): The AWS region name.

        Returns:
            GlueClient: The Glue client.
        """
        return boto3.client("glue", region_name=region_name)

    def database_exists(self) -> bool:
        """Check if a database exists in Glue.

        Returns:
            bool: True if the database exists, False otherwise.
        """
        try:
            self.glue_client.get_database(Name=self.database_name)
            logging.info(f"Database '{self.database_name}' exists.")
            return True
        except self.glue_client.exceptions.EntityNotFoundException:
            logging.info(f"Database '{self.database_name}' does not exist.")
            return False
        except Exception as err:
            logging.error(f"Error checking if database '{self.database_name}' exists: {err}")
            raise

    def create_database(self) -> None:
        """Create a database in Glue."""
        try:
            self.glue_client.create_database(DatabaseInput={"Name": self.database_name})
            logging.info(f"Database '{self.database_name}' created.")
        except self.glue_client.exceptions.AlreadyExistsException:
            logging.warning(f"Database '{self.database_name}' already exists.")
        except Exception as err:
            logging.error(f"Error creating database '{self.database_name}': {err}")

    def table_exists(self) -> bool:
        """Check if a table exists in Glue.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            _ = self.glue_client.get_table(DatabaseName=self.database_name, Name=self.table_name)
            logging.info(f"Table '{self.table_name}' exists.")
            return True
        except self.glue_client.exceptions.EntityNotFoundException:
            logging.info(f"Table '{self.table_name}' does not exist.")
            return False
        except Exception as err:
            logging.error(f"Error checking if table '{self.table_name}' exists: {err}")
            raise

    def create_table_with_flexible_schema(self) -> None:
        """Create a table in Glue with a flexible schema.

        The table has the following columns:
        - project_name: string (partition key)
        - client_key: string (partition key)
        - channel: string (partition key)
        - date: date (partition key)
        - uid: string
        - floats_map: map<string, float>
        - ints_map: map<string, int>
        - strings_map: map<string, string>

        """
        table_input = {
            "Name": self.table_name,
            "StorageDescriptor": {
                "Columns": self.generic_table_columns,
                "Location": self.table_uri,
                "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "SerdeInfo": {
                    "SerializationLibrary": self.serialization_library,
                    "Parameters": {"serialization.format": "1"},
                },
            },
            "PartitionKeys": self.generic_table_partition_keys,
            "TableType": "EXTERNAL_TABLE",
            "Parameters": {"classification": "parquet"},
        }

        try:
            self.glue_client.create_table(
                DatabaseName=self.database_name,
                TableInput=table_input,
            )
            logging.info(f"Table '{self.table_name}' in Database '{self.database_name}' created.")
        except self.glue_client.exceptions.AlreadyExistsException:
            logging.warning(f"Table '{self.table_name}' already exists.")
        except Exception as err:
            logging.error(
                f"Error creating table '{self.table_name}' in Database "
                f"'{self.database_name}': {err}"
            )

    def partition_exists(self) -> bool:
        """Check if a partition exists in Glue.

        Returns:
            bool: True if the partition exists, False otherwise.
        """
        try:
            _ = self.glue_client.get_partition(
                DatabaseName=self.database_name,
                TableName=self.table_name,
                PartitionValues=self.partition_values_list,
            )
            logging.info(f"Partition {self.partition_values_list} exists.")
            return True
        except self.glue_client.exceptions.EntityNotFoundException:
            logging.info(f"Partition {self.partition_values_list} does not exist.")
            return False
        except Exception as err:
            logging.error(f"Error checking if partition exists: {err}")
            raise

    def create_partition(self) -> None:
        """Create a partition for the generic Table & partition schema in Glue.

        The partition has the following columns:
        - project_name: string
        - client_key: string
        - channel: string
        - date: date
        """
        partition_input = {
            "Values": self.partition_values_list,
            "StorageDescriptor": {
                "Columns": self.generic_table_columns,
                "Location": self.partition_uri,
                "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "SerdeInfo": {
                    "SerializationLibrary": self.serialization_library,
                    "Parameters": {"serialization.format": "1"},
                },
            },
        }

        try:
            self.glue_client.create_partition(
                DatabaseName=self.database_name,
                TableName=self.table_name,
                PartitionInput=partition_input,
            )
            logging.info(
                f"Partition added to table '{self.table_name}' in database '{self.database_name}'."
            )
        except self.glue_client.exceptions.AlreadyExistsException:
            logging.warning(f"Partition already exists in table '{self.table_name}'.")
        except Exception as err:
            logging.error(f"Error adding partition to table '{self.table_name}': {err}")

    def delete_partition(self) -> None:
        """Delete a partition from the Glue table."""
        try:
            self.glue_client.delete_partition(
                DatabaseName=self.database_name,
                TableName=self.table_name,
                PartitionValues=self.partition_values_list,
            )
            logging.info(
                f"Partition {self.partition_values_list} deleted from table '{self.table_name}' "
                f"in database '{self.database_name}'."
            )
        except self.glue_client.exceptions.EntityNotFoundException:
            logging.warning(
                f"Partition {self.partition_values_list} does not exist in "
                f"table '{self.table_name}'."
            )
        except Exception as err:
            logging.error(f"Error deleting partition from table '{self.table_name}': {err}")

    def transform_schema_to_map_columns(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Transform the schema of a DataFrame to have map columns.

        The DataFrame is transformed to have the following columns:
        - project_name: string
        - client_key: string
        - channel: string
        - date: date
        - uid: string
        - floats_map: map<string, float>
        - ints_map: map<string, int>
        - strings_map: map<string, string>

        These columns follow the flexible schema approach.

        Args:
            data_df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        # Identify fixed columns
        fixed_columns = self.dataframe_fixed_columns_list

        # Make sure the date column is containing only date values
        data_df["date"] = pd.to_datetime(data_df["date"]).dt.date

        float_columns = data_df.select_dtypes(include=["float32", "float64"]).columns.difference(
            fixed_columns
        )
        int_columns = data_df.select_dtypes(include=["int32", "int64"]).columns.difference(
            fixed_columns
        )
        string_columns = data_df.select_dtypes(include=["object"]).columns.difference(fixed_columns)

        # Create maps for additional columns
        floats_map = data_df[float_columns].apply(
            lambda row: {col: row[col] for col in float_columns if pd.notna(row[col])}, axis=1
        )
        ints_map = data_df[int_columns].apply(
            lambda row: {col: row[col] for col in int_columns if pd.notna(row[col])}, axis=1
        )
        strings_map = data_df[string_columns].apply(
            lambda row: {col: row[col] for col in string_columns if pd.notna(row[col])}, axis=1
        )

        # Create a new DataFrame with the flexible schema
        flexible_df = data_df[fixed_columns].copy()
        flexible_df["floats_map"] = floats_map
        flexible_df["ints_map"] = ints_map
        flexible_df["strings_map"] = strings_map

        return flexible_df

    def upload_dataframe_to_partition(self, data_df: pd.DataFrame) -> None:
        """Upload a DataFrame to a partition in the Athena table."""
        if not self.database_exists():
            self.create_database()

        if not self.table_exists():
            self.create_table_with_flexible_schema()

        if self.partition_exists():
            self.delete_partition()
            s3io.delete_s3_directory(directory_prefix=self.partition_path)

        data_df = self.transform_schema_to_map_columns(data_df)

        arrow_table = arrow_utils.cast_df_as_arrow_table(
            data_df=data_df, schema=self.pyarrow_table_schema
        )

        if arrow_io.upload_pyarrow_table_to_s3(
            table=arrow_table,
            s3_file_path=f"{self.partition_path}/{self.file_name}",
        ):
            self.create_partition()

    @property
    def generic_table_partition_keys(self) -> List[Dict[str, str]]:
        """Property that gets the partition keys for the Athena 'models_monitoring' table.

        These are the keys that are used to partition the data in the Athena table.
        They are to be separated from the table columns.


        Returns:
            List[Dict[str, str]]: The partition keys for the Athena table.
        """
        return [
            {"Name": "project_name", "Type": "string"},
            {"Name": "client_key", "Type": "string"},
            {"Name": "channel", "Type": "string"},
            {"Name": "date", "Type": "date"},
        ]

    @property
    def generic_table_columns(self) -> List[Dict[str, str]]:
        """Property that gets the Athena table columns.

        Returns:
            List[Dict[str, str]]: The Athena table columns

        Raises:
            ValueError: If the table name is not recognized
        """
        if self.table_name == app_state.models_monitoring_table_name:
            return [
                {"Name": "uid", "Type": "string"},
                {"Name": "floats_map", "Type": "map<string,float>"},
                {"Name": "strings_map", "Type": "map<string,string>"},
                {"Name": "ints_map", "Type": "map<string,int>"},
            ]

        if self.table_name == app_state.projects_kpis_table_name:
            return [
                {"Name": "floats_map", "Type": "map<string,float>"},
                {"Name": "strings_map", "Type": "map<string,string>"},
                {"Name": "ints_map", "Type": "map<string,int>"},
            ]

        raise ValueError(f"Table name '{self.table_name}' not recognized.")

    @property
    def dataframe_fixed_columns_list(self) -> List[str]:
        """Property that gets the fixed columns list for the Athena 'models_monitoring' table.

        Returns:
            List[str]: The fixed columns list for the Athena table.
        """
        if self.table_name == app_state.models_monitoring_table_name:
            return ["project_name", "client_key", "channel", "date", "uid"]

        if self.table_name == app_state.projects_kpis_table_name:
            return ["project_name", "client_key", "channel", "date"]

        raise ValueError(f"Table name '{self.table_name}' not recognized.")

    @property
    def pyarrow_table_schema(self) -> pa.Schema:
        """Property that gets the PyArrow schema for the Athena table.

        Returns:
            pa.Schema: The PyArrow schema for the Athena table.
        """
        if self.table_name == app_state.models_monitoring_table_name:
            return pa.schema(
                [
                    ("project_name", pa.string()),
                    ("client_key", pa.string()),
                    ("channel", pa.string()),
                    ("date", pa.date32()),
                    ("uid", pa.string()),
                    ("floats_map", pa.map_(pa.string(), pa.float32())),
                    ("ints_map", pa.map_(pa.string(), pa.int32())),
                    ("strings_map", pa.map_(pa.string(), pa.string())),
                ]
            )

        if self.table_name == app_state.projects_kpis_table_name:
            return pa.schema(
                [
                    ("project_name", pa.string()),
                    ("client_key", pa.string()),
                    ("channel", pa.string()),
                    ("date", pa.date32()),
                    ("floats_map", pa.map_(pa.string(), pa.float32())),
                    ("ints_map", pa.map_(pa.string(), pa.int32())),
                    ("strings_map", pa.map_(pa.string(), pa.string())),
                ]
            )

        raise ValueError(f"Table name '{self.table_name}' not recognized.")
