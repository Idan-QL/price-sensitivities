"""Module of modeling."""

import logging
import multiprocessing
from typing import Tuple

import numpy as np
import pandas as pd

from elasticity.data.configurator import DataColumns
from elasticity.model.cross_validation import cross_validation
from elasticity.model.model import estimate_coefficients
from elasticity.utils.consts import (
    CV_SUFFIXES_CS,
    ELASTIC_THRESHOLD,
    MODEL_SUFFIXES_CS,
    MODEL_TYPES,
    OUTPUT_CS,
    SUPER_ELASTIC_THRESHOLD,
    VERY_ELASTIC_THRESHOLD,
)


def run_model_type(
    data: pd.DataFrame, model_type: str, test_size: float, data_columns: DataColumns
) -> Tuple[dict, float, float]:
    """Calculate cross-validation and regression results.

    Args:
        data (pd.DataFrame): The input data.
        model_type (str): The type of model to run.
        test_size (float): The proportion of the dataset to include in the test split.
        data_columns (DataColumns): Configuration for data columns.

    Returns:
        dict: A dictionary containing the calculated results.
        float: The median quantity from the input data.
        float: The median price from the input data.

    Raises:
        ValueError: If there is an error.
    """
    results = {}
    median_price = data[data_columns.round_price].median()
    median_quantity = data[data_columns.quantity].median()
    try:
        cv_result = cross_validation(
            data=data, model_type=model_type, test_size=test_size, data_columns=data_columns
        )
        estimation_result = estimate_coefficients(
            data=data, model_type=model_type, data_columns=data_columns
        )
        # Store the results in a dictionary
        for suffix in CV_SUFFIXES_CS:
            key = model_type + "_" + suffix
            results[key] = getattr(cv_result, suffix)
        for suffix in MODEL_SUFFIXES_CS:
            key = model_type + "_" + suffix
            results[key] = getattr(estimation_result, suffix)

    except Exception as e:
        logging.info(f"Error in run_model_type: {e}")
        # Set all the results to np.nan
        for col in [model_type + "_" + suffix for suffix in (CV_SUFFIXES_CS + MODEL_SUFFIXES_CS)]:
            results[col] = np.nan
    return results, median_quantity, median_price


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

        thresholds = [q_test_threshold_1, q_test_threshold_2, q_test_threshold_3]
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


# TO DO REVIEW QUALITY TEST
def run_experiment(
    data: pd.DataFrame,
    data_columns: DataColumns,
    test_size: float = 0.1,
    max_pvalue: float = 0.05,
    threshold_best_model_cv_or_refit: int = 7,
    quality_test_error_col: str = "best_relative_absolute_error",
) -> pd.DataFrame:
    """Run experiment and return results DataFrame.

    This function evaluates multiple regression models using cross-validation and
    coefficient estimation. It selects the best model based on the lowest
    'best_model_error_col', given the model's p-value is within the max_pvalue limit.
    It also performs quality tests on 'quality_test_error_col' and generates a
    detailed result.

    Args:
        data (pd.DataFrame): DataFrame containing the dataset.
        data_columns (DataColumns): Configuration for data columns.
        test_size (float, optional): Proportion of the dataset to include in the test split.
        Defaults to 0.1.
        max_pvalue (float, optional): Maximum p-value for model acceptance. Defaults to 0.05.
        threshold_best_model_cv_or_refit (int, optional): threshold to choose best model from
        cv test score or refit. Defaults to 7.
        quality_test_error_col (str, optional): Column name for the quality test error.
        Defaults to "best_relative_absolute_error".

    The function follows these steps:
    1. Initialize a dictionary `results` to store results and variables to track the best model
    and error.
    2. Iterate over each model type in MODEL_TYPES.
       - For each model type, perform cross-validation and coefficient estimation using
       `run_model_type`.
       - Update `results` with the model-specific results.
       - if len(data)>'threshold_best_model_cv_or_refit, best_model_error_col from CV test
       otherwise from refit error
       - Check if the current model has the lowest 'best_model_error_col', meets the p-value
       criterion, and has a non-negative error.
       - If the current model is better than the previous best, update `best_model` and
       `best_error`.
    3. If no model meets the criteria, log the information, assign NaN values for model-specific
    results,
    and set quality test flags to False.
    4. If a best model is found:
       - Assign best model-specific results to the final results.
       - Perform quality tests to determine if the model meets high, medium, or low quality
       standards.
       - Generate a detailed message based on the quality tests and elasticity.
    5. Convert the `results` dictionary to a DataFrame and return it.

    Returns:
        pd.DataFrame: Results of the experiment.
    """
    # Initialize variables
    results = {}
    best_model = None
    best_error = float("inf")  # Initialize with a very large value

    for model_type in MODEL_TYPES:
        model_results, median_quantity, median_price = run_model_type(
            data=data, model_type=model_type, test_size=test_size, data_columns=data_columns
        )

        if len(data) > threshold_best_model_cv_or_refit:
            best_model_error_col = "mean_relative_absolute_error"
        else:
            best_model_error_col = "relative_absolute_error"

        if (
            (model_results[model_type + "_" + best_model_error_col] < best_error)
            and (model_results[model_type + "_pvalue"] <= max_pvalue)
            and (model_results[model_type + "_" + best_model_error_col] >= 0)
        ):
            best_error = model_results[model_type + "_" + best_model_error_col]
            best_model = model_type

        results.update(model_results)

    # If no best model is found, log and assign NaN values
    if best_model is None:
        results["best_model"] = np.nan
        for suffix in MODEL_SUFFIXES_CS:
            results[f"best_{suffix}"] = np.nan
        results["median_quantity"] = np.nan
        results["median_price"] = np.nan

        for col in ["quality_test", "quality_test_high", "quality_test_medium"]:
            results[col] = False

        results["details"] = np.nan
    else:
        results["best_model"] = best_model

        for suffix in MODEL_SUFFIXES_CS:
            results[f"best_{suffix}"] = results[f"{best_model}_{suffix}"]

        results["median_quantity"] = median_quantity
        results["median_price"] = median_price
        results["last_price"] = data["last_price"].iloc[0]
        results["last_date"] = data["last_date"].iloc[0]

        results["quality_test"] = quality_test(
            results["best_elasticity"],
            results["median_quantity"],
            results[quality_test_error_col],
        )
        results["quality_test_high"] = quality_test(
            results["best_elasticity"],
            results["median_quantity"],
            results[quality_test_error_col],
            high_threshold=True,
        )
        results["quality_test_medium"] = (
            results["quality_test"] and not results["quality_test_high"]
        )

        results["elasticity_level"] = make_level(results["best_elasticity"])

        results["details"] = make_details(results["quality_test"], results["quality_test_high"])

    return pd.DataFrame(results, index=[0])


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
            uid=uid, data=df_input, data_columns=data_columns, test_size=test_size
        )
        results_list.append(results_df)
    return pd.concat(results_list)
