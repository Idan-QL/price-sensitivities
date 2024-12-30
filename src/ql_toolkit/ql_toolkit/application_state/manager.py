"""This module contains the Singleton AppState class that holds the state of the application."""

import logging
import os
from typing import ClassVar, Optional


class SingletonMeta(type):
    """SingletonMeta is a metaclass that ensures a class has only one instance.

    It uses a dictionary to store instances of classes that are created using it.
    If a class instance is requested ,and it doesn't exist in the dictionary,
    it's created and stored.
    If it already exists, the stored instance is returned.
    """

    _instances: ClassVar[dict] = {}

    def __call__(cls, *args: list, **kwargs: dict) -> object:
        """This method is called when a class instance is requested.

        It checks if the class instance already exists in the dictionary.
        If it does, it returns the instance.
        If it doesn't, it creates a new instance, stores it in the dictionary, and returns it.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class AppState(metaclass=SingletonMeta):
    """AppState is a singleton class that holds the state of the application.

    It uses the SingletonMeta metaclass to ensure only one instance of AppState exists.
    It contains properties related to the project and bucket names, as well as various S3
    directory paths.

    Attributes:
        run_env (str): The environment the application is running in.
            This value should be obtained as environment variable named 'RUN_ENV'.
            The default value is 'local'. Acceptable values are 'local', 'staging', 'production'.
            This value is typically used to indicate whether the application should connect to a
            production system/DB or to a local system/DB.
        dc_code (str): The data center code. E.g. 'useb', 'usecp' or 'euca'.
        date_format (str): The format of the date string.
        s3_ds_dir (str): The S3 data science directory.
        s3_archive_dir (str): The S3 archive directory.
        s3_db_sets_dir (str): The S3 database sets directory.
        s3_sgmkr_dir (str): The S3 SageMaker directory.
        s3_res_dir (str): The S3 results directory.
        s3_res_attrs_dir (str): The S3 results attributes directory.
        s3_raw_events_dir (str): The S3 raw events' directory.
        attributes_db_sec_name (str): The name of the attributes database section.
        gcp_api_sec_name (str): The name of the Google Cloud Platform API section.

    Properties:
        aws_region (str): The AWS region name.
        project_name (str): The project name.
        bucket_name (str): The bucket name.
        results_type (str): The run type (e.g. 'prod' or 'qa'). This variable would be used to
            indicate whether results should be written to a test DB/directory or not.
    """

    def __init__(self) -> None:
        """Initializes the AppState instance with default values."""
        self._bucket = None
        self._project = None
        self._region = None
        self._results_type = None
        self.run_env = os.getenv("RUN_ENV", "local")
        self.dc_code = os.getenv("DC", "")
        self.date_format = "%Y-%m-%d"
        self.s3_ds_dir = "data_science"
        self.s3_archive_dir = os.path.join(self.s3_ds_dir, "archive")
        self.s3_db_sets_dir = os.path.join(self.s3_archive_dir, "db_sets")
        self.s3_sgmkr_dir = os.path.join(self.s3_ds_dir, "sagemaker")
        self.s3_res_dir = os.path.join(self.s3_sgmkr_dir, "job_results")
        self.s3_res_attrs_dir = "spark/output/actions/"
        self.s3_raw_events_dir = "raw_events/production/product_event/"
        self.attributes_db_sec_name = ""
        self.gcp_api_sec_name = "prod/google_cloud/ds/api_key"

    def __str__(self) -> str:
        """Returns a neatly formatted summary of core AppState attributes."""
        return (
            "AppState:\n"
            f"  Region      : {self.aws_region}\n"
            f"  Bucket      : {self.bucket_name}\n"
            f"  Project     : {self.project_name}\n"
            f"  Run Type    : {self.results_type}\n"
            f"  Environment : {self.run_env}\n"
            f"  DC Code     : {self.dc_code}\n"
        )

    def initialize(
        self, storage_location: str, project: str, is_qa_run: bool, region: Optional[str] = None
    ) -> None:
        """Initializes the AppState instance with the provided values.

        To set the bucket name, pass either the generic storage location name (i.e. 'us' or 'eu')
        or the actual bucket-name to the `storage_location`. This indicates from which storage the
        data will be read (and implies both the bucket and data lake location).

        Args:
            storage_location (str): Bucket name or generic storage location for reading data.
                Must be one of 'us', 'eu', 'quicklizard', or 'quicklizard-eu-central'.
            project (str): The name of the project.
            is_qa_run (bool): If True, sets the run type to 'qa'; otherwise, sets it to 'prod'.
            region (Optional[str]): The region of the bucket. If not provided, the default region
                is set according to the bucket name.

        Returns:
            None

        Raises:
            ValueError: If an unsupported data center or bucket name is provided.

        Examples:
            >>> app_state.initialize(storage_location="us", project="project_name", is_qa_run=True)
            >>> app_state.initialize(
                storage_location="quicklizard-eu-central", project="project_name", is_qa_run=False
                )
        """
        self.bucket_name = storage_location
        self.project_name = project
        self.results_type = "qa" if is_qa_run else "production"

        if region:
            self.aws_region = region

        self._assign_dc_code()
        self._assign_attributes_db_sec_name()

        logging.info(
            f"Initialized AppState with: region = '{self.aws_region}', DC = {self.dc_code}, "
            f"bucket =  '{self.bucket_name}', project = '{self.project_name}' and "
            f"results_type = '{self.results_type}'"
        )

    def _assign_dc_code(self) -> None:
        """Assigns the data center code.

        Assign the data center code based on the run environment and the AWS region.
        It defines the k8s cluster to use for the run. Can be 'useb', 'usecp', 'euca', or 'local'.
        """
        if self.run_env == "production":
            if self.aws_region == "us-east-1":
                self.dc_code = "usecp"
            elif self.aws_region == "eu-central-1":
                self.dc_code = "euca"
        elif self.run_env == "staging" and self.aws_region == "us-east-1":
            self.dc_code = "useb"
        else:
            self.dc_code = "local"

    def _assign_attributes_db_sec_name(self) -> None:
        """Gets the attributes database section name."""
        if self.dc_code != "local":
            self.attributes_db_sec_name = (
                f"rds/internal-attributes-{self.run_env}" f"-{self.dc_code}/rw"
            )

    @property
    def aws_region(self) -> str:
        """Gets the AWS region name.

        Returns:
            (str): The AWS region name.
        """
        if self._region is None:
            err_msg = "Region is not set!"
            raise ValueError(err_msg)
        return self._region

    @aws_region.setter
    def aws_region(self, region: str) -> None:
        """Setter for the AWS region.

        Args:
            region (str): The region to set, must be one of 'us-east-1' or 'eu-central-1'.

        Raises:
            ValueError: If an unsupported region is provided.
        """
        if region in {"us-east-1", "eu-central-1"}:
            self._region = region

    @property
    def project_name(self) -> str:
        """Gets the project name.

        Raises a ValueError if the project name is not set.

        Returns:
            (str): The name of the project.
        """
        if self._project is None:
            err_msg = "Project name is not set!"
            raise ValueError(err_msg)
        return self._project

    @project_name.setter
    def project_name(self, proj_name: str) -> None:
        """Setter for the project name.

        Args:
            proj_name (str): The name of the project.

        Returns:
            None
        """
        self._project = proj_name

    @property
    def bucket_name(self) -> str:
        """Gets the bucket name.

        Returns:
            (str): The name of the bucket.
        """
        if self._bucket is None:
            err_msg = "Bucket is not set!"
            raise ValueError(err_msg)
        return self._bucket

    @bucket_name.setter
    def bucket_name(self, data_center_or_bucket_name: str) -> None:
        """Setter for the bucket name.

        Args:
            data_center_or_bucket_name (str): The data center to set the bucket name for,
                                              or the bucket name itself.

        Raises:
            ValueError: If an unsupported data center or bucket name is provided.

        Examples:
            >>> app_state.bucket_name = "us"
            >>> app_state.bucket_name = "quicklizard-eu-central"
        """
        if data_center_or_bucket_name in {"us", "quicklizard"}:
            self._bucket = "quicklizard"
            self._region = "us-east-1"
        elif data_center_or_bucket_name in {"eu", "quicklizard-eu-central"}:
            self._bucket = "quicklizard-eu-central"
            self._region = "eu-central-1"
        else:
            err_msg = "Unsupported data center or bucket name!"
            raise ValueError(err_msg)

    @property
    def results_type(self) -> str:
        """Gets the run type.

        Returns:
            (str): The run type.
        """
        if self._results_type is None:
            err_msg = "Results type is not set!"
            raise ValueError(err_msg)
        return self._results_type

    @results_type.setter
    def results_type(self, results_type: str) -> None:
        """Setter for the run type.

        Args:
            results_type (str): The run type.

        Returns:
            None
        """
        self._results_type = results_type

    def s3_analytics_dir(self, client_key: str, channel: str) -> str:
        """A function that gets the S3 analytics directory path.

        Args:
            client_key (str): The client key.
            channel (str): The channel.

        Returns:
            (str): The S3 analytics directory path.
        """
        return os.path.join(self.s3_db_sets_dir, client_key, channel, "analytics")

    @property
    def s3_qa_dir(self) -> str:
        """Gets the S3 QA directory path.

        Returns:
            (str): The S3 QA directory path.
        """
        return os.path.join(self.s3_ds_dir, "qa", self.project_name)

    @property
    def s3_artifacts_dir(self) -> str:
        """Gets the S3 artifacts directory path.

        Returns:
            (str): The S3 artifacts directory path.
        """
        return os.path.join(self.s3_sgmkr_dir, "job_artifacts", self.project_name)

    @property
    def s3_conf_dir(self) -> str:
        """Gets the S3 config directory path.

        Returns:
            (str): The S3 config directory path.
        """
        return os.path.join(self.s3_sgmkr_dir, "config_files", self.project_name)

    @property
    def s3_delivery_dir(self) -> str:
        """Gets the S3 delivery directory path.

        Returns:
            (str): The S3 delivery directory path.
        """
        return os.path.join("delivery", self.project_name)

    def s3_datasets_dir(
        self, client_key: Optional[str] = None, channel: Optional[str] = None
    ) -> str:
        """A function that gets the S3 datasets directory path.

        Args:
            client_key (Optional[str]): The client key. Defaults to None.
            channel (Optional[str]): The channel. Defaults to None.

        Returns:
            (str): The S3 datasets directory path.
        """
        if not client_key or not channel:
            return str(os.path.join(self.s3_ds_dir, "datasets"))
        return str(os.path.join(self.s3_ds_dir, "datasets", client_key, channel, self.project_name))

    @property
    def s3_eval_results_dir(self) -> str:
        """Gets the S3 datasets directory path.

        Returns:
            (str): The S3 datasets directory path.
        """
        return str(os.path.join(self.s3_ds_dir, "eval_results", self.project_name))

    @property
    def s3_athena_staging_dir(self) -> str:
        """Gets the S3 athena dir.

        Returns:
            (str): The S3 athena dir.
        """
        if self.bucket_name == "quicklizard":
            return "s3://noam-test-glue-use1/athena/output/"
        if self.bucket_name == "quicklizard-eu-central":
            return "s3://noam-test-glue-euc1/athena/output/"
        return ""

    @property
    def athena_database_name(self) -> str:
        """Gets the Athena database name."""
        if self.results_type == "qa":
            return "ds_sandbox"

        if self.results_type == "production":
            return "ds_production_monitoring"

        err_msg = "Run type is not set!"
        logging.error(err_msg)
        raise ValueError(err_msg)

    @property
    def models_monitoring_table_name(self) -> str:
        """Gets the models monitoring Athena table name."""
        return "models_monitoring"

    @property
    def projects_kpis_table_name(self) -> str:
        """Gets the KPIs Athena table name."""
        return "projects_kpis"

    @property
    def athena_database_s3_uri(self) -> str:
        """Gets the Athena databases directory path."""
        return str(os.path.join(f"s3://{self.bucket_name}", self.s3_ds_dir, "athena_database"))

    @property
    def config_prod_spreadsheet(self) -> str:
        """Gets the name of the prod config spreadsheet.

        Returns:
            (str): The prod config name.
        """
        return "DS_PROD_CONFIG"


app_state = AppState()
