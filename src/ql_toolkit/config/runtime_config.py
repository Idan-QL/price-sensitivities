"""This module contains the Singleton AppState class that holds the state of the application."""

from os import path
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
    """

    def __init__(self) -> None:
        """Initializes the AppState instance with default values."""
        self._bucket = None
        self._project = None
        self.date_format = "%Y-%m-%d"
        self.s3_ds_dir = "data_science"
        self.s3_archive_dir = path.join(self.s3_ds_dir, "archive")
        self.s3_db_sets_dir = path.join(self.s3_archive_dir, "db_sets")
        self.s3_sgmkr_dir = path.join(self.s3_ds_dir, "sagemaker")
        self.s3_res_dir = path.join(self.s3_sgmkr_dir, "job_results")
        self.s3_res_redis_dir = path.join(self.s3_res_dir, "redis_data")
        self.s3_res_bi_ingest_dir = path.join(self.s3_ds_dir, "ingest", "bi_service")
        self.s3_res_attrs_dir = "spark/output/actions/"
        self.s3_python_clients_map_dir = path.join(self.s3_sgmkr_dir, "clients_mappings/")
        self.s3_raw_events_dir = "raw_events/production/product_event/"
        self.analytics_cols = [
            "uid",
            "date",
            "week_of_year",
            "shelf_price",
            "units_sold",
            "inventory",
        ]
        self.db_sets_cols = {}

    @property
    def project_name(self) -> str:
        """Property that gets the project name.

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
        """Property that gets the bucket name.

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

        Returns:
            None

        Example:
            >>> app_state.bucket_name = "us"

        """
        if data_center_or_bucket_name in {"us", "quicklizard"}:
            self._bucket = "quicklizard"
        elif data_center_or_bucket_name in {"eu", "quicklizard-eu-central"}:
            self._bucket = "quicklizard-eu-central"

    @property
    def s3_qa_dir(self) -> str:
        """Property that gets the S3 QA directory path.

        Returns:
            (str): The S3 QA directory path.
        """
        return path.join(self.s3_ds_dir, "qa", self.project_name)

    @property
    def s3_artifacts_dir(self) -> str:
        """Property that gets the S3 artifacts directory path.

        Returns:
            (str): The S3 artifacts directory path.
        """
        return path.join(self.s3_sgmkr_dir, "job_artifacts", self.project_name)

    def s3_monitoring_dir(
        self, client_key: Optional[str] = None, channel: Optional[str] = None
    ) -> str:
        """Property that gets the S3 monitoring directory path.

        Args:
            client_key (Optional[str]): The client key. Defaults to None.
            channel (Optional[str]): The channel. Defaults to None.

        Returns:
            (str): The S3 monitoring directory path.
        """
        if client_key is None or channel is None:
            return path.join(self.s3_sgmkr_dir, "services_monitoring", self.project_name)
        if client_key is not None and channel is not None:
            return path.join(
                self.s3_sgmkr_dir,
                "services_monitoring",
                self.project_name,
                client_key,
                channel,
            )
        return ""

    @property
    def s3_conf_dir(self) -> str:
        """Property that gets the S3 config directory path.

        Returns:
            (str): The S3 config directory path.
        """
        return path.join(self.s3_sgmkr_dir, "config_files", self.project_name)

    @property
    def s3_delivery_dir(self) -> str:
        """Property that gets the S3 delivery directory path.

        Returns:
            (str): The S3 delivery directory path.
        """
        return path.join("delivery", self.project_name)

    @property
    def s3_athena_staging_dir(self) -> str:
        """Property that gets the S3 athena dir.

        Returns:
            (str): The S3 athena dir.
        """
        if self.bucket_name == "quicklizard":
            return "s3://noam-test-glue-use1/athena/output/"
        if self.bucket_name == "quicklizard-eu-central":
            return "s3://noam-test-glue-euc1/athena/output/"
        return ""

    @property
    def s3_region(self) -> str:
        """Property that gets the S3 athena dir.

        Returns:
            (str): The S3 athena dir.
        """
        if self.bucket_name == "quicklizard":
            return "us-east-1"
        if self.bucket_name == "quicklizard-eu-central":
            return "eu-central-1"
        return ""

    def s3_datasets_dir(
        self, client_key: Optional[str] = None, channel: Optional[str] = None
    ) -> str:
        """Property that gets the S3 datasets directory path.

        Args:
            client_key (Optional[str]): The client key. Defaults to None.
            channel (Optional[str]): The channel. Defaults to None.

        Returns:
            (str): The S3 datasets directory path.
        """
        if not client_key or not channel:
            return str(path.join(self.s3_ds_dir, "datasets"))
        return str(path.join(self.s3_ds_dir, "datasets", client_key, channel, self.project_name))

    @property
    def s3_eval_results_dir(self) -> str:
        """Property that gets the S3 eval result path.

        Returns:
            (str): The S3 eval result path.
        """
        return str(path.join(self.s3_ds_dir, "eval_results", self.project_name))

    @property
    def athena_databases_dir(self) -> str:
        """Property that gets the Athena databases directory path."""
        return str(path.join(self.s3_ds_dir, "athena_databases", self.project_name))

    @property
    def s3_config_directory(self) -> str:
        """Property that gets the S3 config dir.

        Returns:
            (str): The S3 config dir.
        """
        return "data_science/credentials/"

    @property
    def config_prod_spreadsheet(self) -> str:
        """Property that gets the name of the prod config spreadsheet.

        Returns:
            (str): The prod config name.
        """
        return "DS_PROD_CONFIG"

    @property
    def config_qa_spreadsheet(self) -> str:
        """Property that gets the name of the QA config spreadsheet.

        Returns:
            (str): The qa config name.
        """
        return "DS_QA_CONFIG"


app_state = AppState()
