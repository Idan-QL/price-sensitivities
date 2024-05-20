"""Module of modeling."""

import logging
import multiprocessing
from typing import Tuple

import numpy as np
import pandas as pd

from elasticity.model.cross_validation import cross_validation
from elasticity.model.model import estimate_coefficients
from elasticity.utils.consts import (
    CV_SUFFIXES_CS,
    MODEL_SUFFIXES_CS,
    MODEL_TYPES,
    OUTPUT_CS,
)


def run_model_type(
    data: pd.DataFrame,
    model_type: str,
    test_size: float,
    price_col: str,
    quantity_col: str,
    weights_col: str,
) -> Tuple[dict, float, float]:
    """Calculate cross-validation and regression results.

    Args:
        data (pd.DataFrame): The input data.
        model_type (str): The type of model to run.
        test_size (float): The proportion of the dataset to include in the test split.
        price_col (str): The name of the column containing the prices.
        quantity_col (str): The name of the column containing the quantities.
        weights_col (str): The name of the column containing the weights.

    Returns:
        dict: A dictionary containing the calculated results.
        float: The median quantity from the input data.
        float: The median price from the input data.

    Raises:
        ValueError: If there is an error.
    """
    results = {}
    median_price = data[price_col].median()
    median_quantity = data[quantity_col].median()
    try:
        cv_result = cross_validation(
            data, model_type, test_size, price_col, quantity_col, weights_col
        )
        estimation_result = estimate_coefficients(
            data,
            model_type,
            price_col=price_col,
            quantity_col=quantity_col,
            weights_col=weights_col,
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
        for col in [
            model_type + "_" + suffix for suffix in (CV_SUFFIXES_CS + MODEL_SUFFIXES_CS)
        ]:
            results[col] = np.nan
    return results, median_quantity, median_price


def quality_test(
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

    This function performs a quality test based on the provided median quantity and
    relative absolute error. The test checks if the median quantity and relative absolute
    error meet certain thresholds based on the provided parameters.

    The thresholds can be adjusted by modifying the function parameters. If `high_threshold`
    is True, thresholds are halved, making the test stricter. For example, if the original
    thresholds list is [50, 40, 30], then if `high_threshold` is True, the new thresholds
    list will be [25, 20, 15]. Thus, the high quality test is twice as strict as the medium
    quality test.

    Args:
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
        if not isinstance(median_quantity, (int, float)):
            raise ValueError("median_quantity must be a number.")
        if not isinstance(best_relative_absolute_error, (int, float)):
            raise ValueError("best_relative_absolute_error must be a number.")
        thresholds = [q_test_threshold_1, q_test_threshold_2, q_test_threshold_3]
        if high_threshold:
            thresholds = [threshold // 2 for threshold in thresholds]
        return (
            (
                median_quantity < q_test_value_1
                and best_relative_absolute_error <= thresholds[0]
            )
            or (
                q_test_value_1 <= median_quantity < q_test_value_2
                and best_relative_absolute_error <= thresholds[1]
            )
            or (
                median_quantity >= q_test_value_2
                and best_relative_absolute_error <= thresholds[2]
            )
        )
    except ValueError as e:
        if high_threshold:
            logging.error(f"High quality_test failed,: {e}")
        else:
            logging.error(f"Medium quality_test failed,: {e}")
        return False


def make_details(
    is_quality_test_passed: bool, is_high_quality_test_passed: bool, elasticity: float
) -> str:
    """Generate a concise message based on the quality test and elasticity.

    Args:
        is_quality_test_passed (bool): Indicates if a quality test was conducted.
        is_high_quality_test_passed (bool): Indicates if the quality test was of high quality.
        elasticity (float): The elasticity value.

    Returns:
        str: A concise message describing the elasticity and quality test conclusion.
    """
    # Determine elasticity description
    if elasticity < -2.5:
        elasticity_message = "Very elastic"
    elif elasticity < -1:
        elasticity_message = "Elastic"
    elif elasticity < 0:
        elasticity_message = "Inelastic"
    elif elasticity > 0:
        elasticity_message = "Positively elastic"
    else:
        elasticity_message = ""

    # Determine quality test conclusion
    if is_quality_test_passed and is_high_quality_test_passed:
        quality_test_message = "High quality test"
    elif is_quality_test_passed and not is_high_quality_test_passed:
        quality_test_message = "Medium quality test"
    elif not is_quality_test_passed and not is_high_quality_test_passed:
        quality_test_message = "Low quality test"
    else:
        quality_test_message = ""

    return f"{elasticity_message} - {quality_test_message}."


# TO DO REVIEW QUALITY TEST
def run_experiment(
    data: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
    max_pvalue: float = 0.05,
    best_model_error_col: str = "mean_relative_absolute_error",
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
        test_size (float, optional): Proportion of the dataset to include in the test split.
        Defaults to 0.1.
        price_col (str, optional): Column name for prices. Defaults to "price".
        quantity_col (str, optional): Column name for quantities. Defaults to "quantity".
        weights_col (str, optional): Column name for weights. Defaults to "days".
        max_pvalue (float, optional): Maximum p-value for model acceptance. Defaults to 0.05.
        best_model_error_col (str, optional): Column name for the best model error.
        Defaults to "relative_absolute_error".
        quality_test_error_col (str, optional): Column name for the quality test error.
        Defaults to "best_relative_absolute_error".

    The function follows these steps:
    1. Initialize a dictionary `results` to store results and variables to track the best model
    and error.
    2. Iterate over each model type in MODEL_TYPES.
       - For each model type, perform cross-validation and coefficient estimation using
       `run_model_type`.
       - Update `results` with the model-specific results.
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
            data, model_type, test_size, price_col, quantity_col, weights_col
        )

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

        results["quality_test"] = quality_test(
            results["median_quantity"], results[quality_test_error_col]
        )
        results["quality_test_high"] = quality_test(
            results["median_quantity"],
            results[quality_test_error_col],
            high_threshold=True,
        )
        results["quality_test_medium"] = (
            results["quality_test"] and not results["quality_test_high"]
        )

        results["details"] = make_details(
            results["quality_test"],
            results["quality_test_high"],
            results["power_elasticity"],
        )

    return pd.DataFrame(results, index=[0])


def run_experiment_for_uid(
    uid: str,
    data: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
) -> pd.DataFrame:
    """Run experiment for a specific user ID.

    Args:
        uid (str): The user ID for which the experiment is run.
        data (pd.DataFrame): The input data containing the user's data.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.1.
        price_col (str, optional): The column name for the price data. Defaults to "price".
        quantity_col (str, optional): The column name for the quantity data. Defaults to "quantity".
        weights_col (str, optional): The column name for the weights data. Defaults to "days".

    Returns:
        pd.DataFrame: The results of the experiment for the specified user ID.
    """
    subset_data = data[data["uid"] == uid]
    try:
        results_df = run_experiment(
            data=subset_data,
            test_size=test_size,
            price_col=price_col,
            quantity_col=quantity_col,
            weights_col=weights_col,
        )
    except Exception as e:
        logging.info(f"Error for UID {uid}: {e}")

        results_df = pd.DataFrame(np.nan, index=[0], columns=OUTPUT_CS)
    results_df["uid"] = uid
    return results_df


def run_experiment_for_uids_parallel(
    df_input: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
) -> pd.DataFrame:
    """Run experiment for multiple UID in parallel.

    Args:
        df_input (pd.DataFrame): The input DataFrame containing the data.
        test_size (float, optional): The proportion of the data to use for testing. Defaults to 0.1.
        price_col (str, optional): The name of the column containing the price data.
        Defaults to "price".
        quantity_col (str, optional): The name of the column containing the quantity data.
        Defaults to "quantity".
        weights_col (str, optional): The name of the column containing the weights data.
        Defaults to "days".

    Returns:
        pd.DataFrame: The concatenated DataFrame containing the results of the experiments
        for each user ID.
    """
    # Delete rows with price equal to zero
    df_input = df_input[df_input[price_col] != 0]
    unique_uids = df_input["uid"].unique()
    total_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool((total_cores - 1))
    results_list = pool.starmap(
        run_experiment_for_uid,
        [
            (uid, df_input, test_size, price_col, quantity_col, weights_col)
            for uid in unique_uids
        ],
    )
    pool.close()
    pool.join()
    return pd.concat(results_list)


def run_experiment_for_uids_not_parallel(
    df_input: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
) -> pd.DataFrame:
    """Run experiment for multiple user IDs (not in parallel).

    Args:
        df_input (pd.DataFrame): The input DataFrame containing the data.
        test_size (float, optional): The proportion of the data to use for testing.
        Defaults to 0.1.
        price_col (str, optional): The name of the column containing the price data.
        Defaults to "price".
        quantity_col (str, optional): The name of the column containing the quantity data.
        Defaults to "quantity".
        weights_col (str, optional): The name of the column containing the weights data.
        Defaults to "days".

    Returns:
        pd.DataFrame: The concatenated DataFrame of results for each user ID.
    """
    # Delete rows with price equal to zero
    df_input = df_input[df_input[price_col] != 0]
    results_list = []
    unique_uids = df_input["uid"].unique()
    for uid in unique_uids:
        results_df = run_experiment_for_uid(
            uid, df_input, test_size, price_col, quantity_col, weights_col
        )
        results_list.append(results_df)
    return pd.concat(results_list)
