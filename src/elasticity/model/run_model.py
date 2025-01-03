"""Module of modeling."""

import logging
import multiprocessing

import numpy as np
import pandas as pd
from pydantic import BaseModel

from elasticity.data.configurator import DataColumns
from elasticity.model.cross_validation import CrossValidationResult, cross_validation
from elasticity.model.model import EstimationResult, estimate_coefficients
from elasticity.utils.consts import (
    CV_SUFFIXES_CS,
    ELASTIC_THRESHOLD,
    MODEL_SUFFIXES_CS,
    MODEL_TYPES,
    OUTPUT_CS,
    SUPER_ELASTIC_THRESHOLD,
    VERY_ELASTIC_THRESHOLD,
)


class ModelResults:
    """A class to store and manage results for a single model type.

    This class encapsulates the results obtained from both cross-validation and
    estimation phases of a model's computation.

    Attributes:
        model_type (str): The type of model for which results are stored.
        results (dict): A dictionary to store various model metrics.
    """

    def __init__(self, model_type: str) -> None:
        """Initializes a new instance of the ModelResults class.

        Args:
            model_type (str): The type of model for which results are stored.
        """
        self.model_type = model_type
        self.results = {}

    def add_cv_result(self, cv_result: CrossValidationResult) -> None:
        """Adds cross-validation results to the results dictionary.

        Args:
            cv_result (CrossValidationResult): The cross-validation results object containing
            metrics.
        """
        for suffix in CV_SUFFIXES_CS:
            key = f"{self.model_type}_{suffix}"
            self.results[key] = getattr(cv_result, suffix)

    def add_estimation_result(self, estimation_result: EstimationResult) -> None:
        """Adds estimation results to the results dictionary.

        Args:
            estimation_result (EstimationResult): The estimation results object containing metrics.
        """
        for suffix in MODEL_SUFFIXES_CS:
            key = f"{self.model_type}_{suffix}"
            self.results[key] = getattr(estimation_result, suffix)

    def clear_results(self) -> None:
        """Clears all fields in the `results` dictionary by setting them to NaN."""
        for suffix in CV_SUFFIXES_CS + MODEL_SUFFIXES_CS:
            self.results[f"{self.model_type}_{suffix}"] = np.nan

    def get_results(self) -> dict:
        """Retrieves all stored results.

        Returns:
            dict: `results` attribute containing all results stored for the model.
        """
        return self.results


class ExperimentConfig(BaseModel):
    """Configuration settings for evaluating and selecting the best model in an experiment.

    Attributes:
        best_model_error_type (str): The name of error type to use for determining the best model.
        max_pvalue (float): The maximum p-value threshold for considering a model as potentially
        best.
    """

    best_model_error_type: str
    max_pvalue: float


class DataConfig(BaseModel):
    """Configuration settings containing data-related parameters for an experiment.

    These settings are used to provide contextual data which might be relevant for the experiment's
    logic and outputs.

    Attributes:
        median_quantity (float): The median quantity value from the dataset.
        median_price (float): The median price value from the dataset.
        last_price (float): The last recorded price value from the dataset.
        last_date (str): The date corresponding to the last data entry.
    """

    median_quantity: float
    median_price: float
    last_price: float
    last_date: str


class ExperimentResults:
    """A class to store results of an experiment and determine the best model.

    This class manages the integration of models' results and configuration settings to identify
    and record the best-performing model according to specified performance metrics and thresholds.

    Attributes:
        results (dict): A dictionary to store experiment results and metadata.
        experiment_config (ExperimentConfig): Configuration settings for experiment.
        data_config (DataConfig): Configuration settings for data-specific parameters.
    """

    def __init__(self, experiment_config: ExperimentConfig, data_config: DataConfig) -> None:
        """Initializes a new instance of the ExperimentResults class.

        Args:
            experiment_config (ExperimentConfig): Configuration settings for experiment.
            data_config (DataConfig): Configuration settings for data-specific parameters.
        """
        self.results = {}
        self.results["best_model"] = None
        self.experiment_config = experiment_config
        self.data_config = data_config

    def add_model_results(self, model_results: ModelResults) -> None:
        """Integrates results from a single model into the experiment `results` dictionary.

        Args:
            model_results (ModelResults): ModelResults object containing data from a single model
            run.
        """
        self.results.update(model_results.get_results())

    def find_best_model(self) -> None:
        """Identifies the best model based on pre-configured error type and p-value threshold.

        This method iterates through each model type defined in the global `MODEL_TYPES`.
        For each model, it retrieves the value of `best_model_error_type` and p-value from
        `results` attribute. It then checks if the current model has the lowest value of
        `best_model_error_type`, meets the p-value criterion, and has a non-negative error.
        Once the best model is identified, the method updates the `results` dictionary.

        If required keys (related to `best_model_error_type` or p-value) are not found in the
        `results` for some model, this will be logged as error. Such model will not be
        considered as potentially best model. This can happen if model's results are not
        added to the ExperimentResults object (using `add_model_results` method).
        """
        best_error = float("inf")
        for model_type in MODEL_TYPES:
            model_error_key = f"{model_type}_{self.experiment_config.best_model_error_type}"
            model_pvalue_key = f"{model_type}_pvalue"
            if model_error_key not in self.results or model_pvalue_key not in self.results:
                # If required keys are not found, log as error and skip this model
                err_msg = f"""Required keys not found in `results` for the {model_type} model.\n
                Ensure that all model's results have been added to the ExperimentResults object\n
                (using `add_model_results` method)."""
                logging.error(err_msg)
                continue
            model_error = self.results[model_error_key]
            model_pvalue = self.results[model_pvalue_key]

            if (
                model_error < best_error
                and model_error >= 0
                and model_pvalue <= self.experiment_config.max_pvalue
            ):
                self.results["best_model"] = model_type
                best_error = model_error

    def compute_results(self) -> None:
        """Sets the best model for the experiment and populates the `results` dictionary.

        This method first identifies the best model using the `find_best_model` method.
        If the best model is identified, the method runs the `populate_results_for_best_model`,
        otherwise it runs the `populate_results_for_no_best_model` method. Both methods populate
        the `results` dictionary with the relevant data based on the best model or lack thereof.
        """
        self.find_best_model()
        if self.results["best_model"] is None:
            self._populate_results_for_no_best_model()
        else:
            self._populate_results_for_best_model()

    def _populate_results_for_no_best_model(self) -> None:
        """Populates the `results` with NaN or False values when no best model is found."""
        for suffix in MODEL_SUFFIXES_CS:
            self.results[f"best_{suffix}"] = np.nan
        # Add all fields from data_config with NaN values to results
        self.results.update(dict.fromkeys(self.data_config.model_dump(), np.nan))
        quality_tests = {
            "quality_test": False,
            "quality_test_high": False,
            "quality_test_medium": False,
        }
        self.results.update(quality_tests)
        self.results["details"] = np.nan
        self.results["elasticity_level"] = np.nan

    def _populate_results_for_best_model(self) -> None:
        """Populates the `results` dictionary when the best model is found.

        This method populates the `results` dictionary with data specific to the best model,
        such as its error metrics and other relevant details. It also copies the data from the
        `data_config` attribute into the `results` dictionary. In addition, it conducts quality
        tests (to check if the best model meets high, medium, or low quality standards) and creates
        detailed text messages based on the quality tests outcomes and elasticity level.
        """
        for suffix in MODEL_SUFFIXES_CS:
            self.results[f"best_{suffix}"] = self.results[f"{self.results['best_model']}_{suffix}"]
        # Add all fields from data_config to results
        self.results.update(self.data_config.model_dump())
        # Add fields for quality tests (quality_test, quality_test_high, quality_test_medium)
        self._run_quality_tests()
        # Add text messages for details related to quality tests and elasticity level
        self._make_details_and_level()

    def _make_details_and_level(self) -> None:
        """Updates the `results` with detailed messages on quality tests and elasticity level."""
        self.results["details"] = make_details(
            self.results["quality_test"], self.results["quality_test_high"]
        )
        self.results["elasticity_level"] = make_level(self.results["best_elasticity"])

    def _run_quality_tests(self) -> None:
        """Conducts quality tests for the best model and updates the `results` dictionary.

        This method performs quality tests based on the elasticity, median quantity, and
        relative absolute error of the best model. It updates the `results` dictionary with the
        information whether the model passes the (low) quality test, high quality test,
        and medium quality test.
        """
        elasticity = self.results["best_elasticity"]
        median_quantity = self.data_config.median_quantity
        best_relative_absolute_error = self.results["best_relative_absolute_error"]
        quality_tests = {}
        quality_tests["quality_test"] = quality_test(
            elasticity, median_quantity, best_relative_absolute_error
        )
        quality_tests["quality_test_high"] = quality_test(
            elasticity,
            median_quantity,
            best_relative_absolute_error,
            high_threshold=True,
        )
        quality_tests["quality_test_medium"] = (
            quality_tests["quality_test"] and not quality_tests["quality_test_high"]
        )
        self.results.update(quality_tests)

    def clear_results(self) -> None:
        """Clears all fields in the `results` dictionary by setting them to NaN."""
        for suffix in OUTPUT_CS:
            self.results[f"{suffix}"] = np.nan

    def get_results(self) -> dict:
        """Retrieves all stored results.

        Returns:
            dict: `results` attribute containing all results stored for the experiment.
        """
        return self.results


def run_model_type(
    data: pd.DataFrame,
    data_columns: DataColumns,
    model_type: str,
    test_size: float,
) -> ModelResults:
    """Calculate cross-validation and regression results for specified `model_type`.

    This function calculates the cross-validation and regression results for a specified model type
    and saves the results in a ModelResults object. If an exception occurs during the calculation,
    the function logs the error and returns an ModelResults object with NaN values.

    Args:
        data (pd.DataFrame): The input data.
        data_columns (DataColumns): Configuration for data columns.
        model_type (str): The type of model to run.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        ModelResults: An object containing the calculated results.
    """
    model_results = ModelResults(model_type)
    try:
        cv_result = cross_validation(
            data=data,
            model_type=model_type,
            test_size=test_size,
            data_columns=data_columns,
        )
        estimation_result = estimate_coefficients(
            data=data, model_type=model_type, data_columns=data_columns
        )
        model_results.add_cv_result(cv_result)
        model_results.add_estimation_result(estimation_result)
    except Exception as e:
        # If exception occurs, log the error and and return empty results
        logging.error(f"Error in run_model_type for {model_type}: {e}")
        model_results.clear_results()
    return model_results


def quality_test(
    elasticity: float,
    median_quantity: float,
    best_relative_absolute_error: float,
    high_threshold: bool = False,
    q_test_value_1: float = 30,
    q_test_value_2: float = 100,
    q_test_threshold_1: float = 50,
    q_test_threshold_2: float = 40,
    q_test_threshold_3: float = 30,
) -> bool:
    """Perform quality test based on median quantity and relative absolute error.

    If Elasticty is negative or equal to 0, this function performs a quality test based on
    the provided median quantity and relative absolute error.
    The test checks if the median quantity and relative absolute error meet certain
    thresholds based on the provided parameters.

    The thresholds can be adjusted by modifying the function parameters. If `high_threshold`
    is True, thresholds are halved, making the test stricter. For example, if the original
    thresholds list is [50, 40, 30], then if `high_threshold` is True, the new thresholds
    list will be [25, 20, 15]. Thus, the high quality test is twice as strict as the medium
    quality test.

    Args:
        elasticity (float): The Elasticity
        median_quantity (float): The median quantity value.
        best_relative_absolute_error (float): The relative absolute error of the best model.
        high_threshold (bool): If True, apply high threshold by halving the thresholds.
        q_test_value_1 (float): Value for the first quality test threshold.
        q_test_value_2 (float): Value for the second quality test threshold.
        q_test_threshold_1 (float): Threshold for the first quality test.
        q_test_threshold_2 (float): Threshold for the second quality test.
        q_test_threshold_3 (float): Threshold for the third quality test.

    Returns:
        bool: True if the test passes, False otherwise.

    Raises:
        ValueError: If input values are not of the expected type.
    """
    try:
        if not isinstance(elasticity, (int, float)):
            raise ValueError("elasticity must be a number.")
        if not isinstance(median_quantity, (int, float)):
            raise ValueError("median_quantity must be a number.")
        if not isinstance(best_relative_absolute_error, (int, float)):
            raise ValueError("best_relative_absolute_error must be a number.")

        if elasticity > 0:
            return False

        thresholds = [
            q_test_threshold_1,
            q_test_threshold_2,
            q_test_threshold_3,
        ]
        if high_threshold:
            thresholds = [threshold // 2 for threshold in thresholds]
        return (
            (median_quantity < q_test_value_1 and best_relative_absolute_error <= thresholds[0])
            or (
                q_test_value_1 <= median_quantity < q_test_value_2
                and best_relative_absolute_error <= thresholds[1]
            )
            or (median_quantity >= q_test_value_2 and best_relative_absolute_error <= thresholds[2])
        )
    except ValueError as e:
        if high_threshold:
            logging.error(f"High quality_test failed,: {e}")
        else:
            logging.error(f"Medium quality_test failed,: {e}")
        return False


def make_level(elasticity: float) -> str:
    """Generate a concise message based on the quality test and elasticity.

    Args:
        elasticity (float): The elasticity value.

    Returns:
        str: A concise message describing the elasticity level.
    """
    # Determine elasticity description
    if elasticity < SUPER_ELASTIC_THRESHOLD:
        elasticity_level = "Super elastic"
    elif elasticity < VERY_ELASTIC_THRESHOLD:
        elasticity_level = "Very elastic"
    elif elasticity < ELASTIC_THRESHOLD:
        elasticity_level = "Elastic"
    elif elasticity < 0:
        elasticity_level = "Inelastic"
    elif elasticity > 0:
        elasticity_level = "Positively elastic"
    else:
        elasticity_level = ""

    return f"{elasticity_level}"


def make_details(is_quality_test_passed: bool, is_high_quality_test_passed: bool) -> str:
    """Generate a concise message based on the quality test and elasticity.

    Args:
        is_quality_test_passed (bool): Indicates if a quality test was conducted.
        is_high_quality_test_passed (bool): Indicates if the quality test was of high quality.

    Returns:
        str: A concise message describing the quality test conclusion.
    """
    # Determine quality test conclusion
    if is_quality_test_passed and is_high_quality_test_passed:
        quality_test_message = "High quality test"
    elif is_quality_test_passed and not is_high_quality_test_passed:
        quality_test_message = "Medium quality test"
    elif not is_quality_test_passed and not is_high_quality_test_passed:
        quality_test_message = "Low quality test"
    else:
        quality_test_message = ""

    return f"{quality_test_message}."


def run_experiment(
    data: pd.DataFrame,
    data_columns: DataColumns,
    test_size: float = 0.1,
    max_pvalue: float = 0.05,
    threshold_best_model_cv_or_refit: int = 7,
) -> pd.DataFrame:
    """Run experiment and return results in a DataFrame.

    This function evaluates multiple regression models using cross-validation and
    coefficient estimation. It selects the best model based on the lowest
    `best_model_error_type`, given the model's p-value is within the max_pvalue limit.
    It also performs quality tests for the best model and generates a detailed result.
    If the experiment fails, it logs the error and returns a DataFrame with NaN values
    for all expected output columns.

    Args:
        data (pd.DataFrame): The input data.
        data_columns (DataColumns): Configuration for data columns.
        test_size (float, optional): The proportion of the dataset to include in the test split.
        Defaults to 0.1.
        max_pvalue (float, optional): Maximum p-value for model acceptance. Defaults to 0.05.
        threshold_best_model_cv_or_refit (int, optional): threshold to choose best model from
        cv test score or refit. Defaults to 7.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated results of the experiment, with one
        row and various columns for each result metric and model type.
    """
    best_model_error_type = (
        "mean_relative_absolute_error"
        if len(data) > threshold_best_model_cv_or_refit
        else "relative_absolute_error"
    )
    median_quantity = data[data_columns.quantity].median()
    median_price = data[data_columns.round_price].median()
    last_price = data["last_price"].iloc[0]
    last_date = data["last_date"].iloc[0]
    data_config = DataConfig(
        median_quantity=median_quantity,
        median_price=median_price,
        last_price=last_price,
        last_date=last_date,
    )
    experiment_config = ExperimentConfig(
        best_model_error_type=best_model_error_type, max_pvalue=max_pvalue
    )
    experiment_results = ExperimentResults(
        experiment_config=experiment_config, data_config=data_config
    )
    try:
        for model_type in MODEL_TYPES:
            model_results = run_model_type(data, data_columns, model_type, test_size)
            experiment_results.add_model_results(model_results)
        experiment_results.compute_results()
    except Exception as e:
        # If exception occurs, log the error and clear the results
        logging.error(f"Error in run_experiment: {e}")
        experiment_results.clear_results()

    return pd.DataFrame(experiment_results.get_results(), index=[0])


def run_experiment_for_uid(
    uid: str,
    data: pd.DataFrame,
    data_columns: DataColumns,
    test_size: float = 0.1,
) -> pd.DataFrame:
    """Run experiment for a specific user ID.

    Args:
        uid (str): The user ID for which the experiment is run.
        data (pd.DataFrame): The input data containing the user's data.
        data_columns (DataColumns): Configuration for data columns.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.1.

    Returns:
        pd.DataFrame: The results of the experiment for the specified user ID.
    """
    subset_data = data[data[data_columns.uid] == uid]
    try:
        results_df = run_experiment(
            data=subset_data,
            data_columns=data_columns,
            test_size=test_size,
        )
    except Exception as e:
        logging.info(f"Error for UID {uid}: {e}")

        results_df = pd.DataFrame(np.nan, index=[0], columns=OUTPUT_CS)
    results_df[data_columns.uid] = uid
    return results_df


def run_experiment_for_uids_parallel(
    df_input: pd.DataFrame, data_columns: DataColumns, test_size: float = 0.1
) -> pd.DataFrame:
    """Run experiment for multiple UID in parallel.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing the data.
        data_columns (DataColumns): Configuration for data columns.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.1.

    Returns:
        pd.DataFrame: The concatenated DataFrame containing the results of the experiments
        for each user ID.
    """
    # Delete rows with price equal to zero
    df_input = df_input[df_input[data_columns.round_price] != 0]
    unique_uids = df_input[data_columns.uid].unique()
    total_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(total_cores - 1) as pool:
        results_list = pool.starmap(
            run_experiment_for_uid,
            [(uid, df_input, data_columns, test_size) for uid in unique_uids],
        )
    return pd.concat(results_list)


def run_experiment_for_uids_not_parallel(
    df_input: pd.DataFrame, data_columns: DataColumns, test_size: float = 0.1
) -> pd.DataFrame:
    """Run experiment for multiple user IDs (not in parallel).

    Args:
        df_input (pd.DataFrame): The input DataFrame containing the data.
        data_columns (DataColumns): Configuration for data columns.
        test_size (float, optional): The proportion of the data to use for testing.
        Defaults to 0.1.

    Returns:
        pd.DataFrame: The concatenated DataFrame of results for each user ID.
    """
    # Delete rows with price equal to zero
    df_input = df_input[df_input[data_columns.round_price] != 0]
    results_list = []
    unique_uids = df_input[data_columns.uid].unique()
    for uid in unique_uids:
        results_df = run_experiment_for_uid(
            uid=uid,
            data=df_input,
            data_columns=data_columns,
            test_size=test_size,
        )
        results_list.append(results_df)
    return pd.concat(results_list)
