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
    best_mean_relative_error: float,
    high_threshold: bool = False,
    q_test_value_1: float = 30,
    q_test_value_2: float = 100,
    q_test_threshold_1: float = 50,
    q_test_threshold_2: float = 40,
    q_test_threshold_3: float = 30,
) -> bool:
    """Perform quality test based on median quantity and mean relative error.

    This function performs a quality test based on the provided median quantity and
    mean relative error. The test checks if the median quantity and mean relative
    error meet certain thresholds based on the provided parameters.
    The thresholds can be adjusted by modifying the function parameters.

    Args:
        median_quantity: median quantity value
        best_mean_relative_error: mean relative error of the best model
        high_threshold: if True, apply high threshold, otherwise apply regular threshold
        q_test_value_1: value for first quality test threshold
        q_test_value_2: value for second quality test threshold
        q_test_threshold_1: threshold for the first quality test
        q_test_threshold_2: threshold for the second quality test
        q_test_threshold_3: threshold for the third quality test

    Returns:
        bool: True if the test passes, False otherwise
    """
    thresholds = [q_test_threshold_1, q_test_threshold_2, q_test_threshold_3]
    if high_threshold:
        thresholds = [threshold // 2 for threshold in thresholds]
    return (
        (median_quantity < q_test_value_1 and best_mean_relative_error <= thresholds[0])
        or (
            q_test_value_1 <= median_quantity < q_test_value_2
            and best_mean_relative_error <= thresholds[1]
        )
        or (
            median_quantity >= q_test_value_2
            and best_mean_relative_error <= thresholds[2]
        )
    )


def make_details(quality_test: bool, quality_test_high: bool, elasticity: float) -> str:
    """Generate a concise message based on the quality test and elasticity.

    Args:
        quality_test (bool): Indicates if a quality test was conducted.
        quality_test_high (bool): Indicates if the quality test was of high quality.
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
    if quality_test and quality_test_high:
        quality_test_message = "High quality test"
    elif quality_test and not quality_test_high:
        quality_test_message = "Medium quality test"
    elif not quality_test and not quality_test_high:
        quality_test_message = "Low quality test"
    else:
        quality_test_message = (
            ""  # Adjust according to your needs if other conditions exist
        )

    return f"{elasticity_message} - {quality_test_message}."


# TO DO REVIEW QUALITY TEST
def run_experiment(
    data: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
    max_pvalue: float = 0.05,
    best_model_error_col: str = "relative_absolute_error",
    quality_test_error_col: str = "best_relative_absolute_error",
) -> pd.DataFrame:
    """Run experiment and return results DataFrame.

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

    Returns:
        pd.DataFrame: Results of the experiment.
    """
    # Initialize variables
    results = {}
    best_model = None
    best_error = float("inf")  # Initialize with a very large value

    # Iterate over different model types
    for model_type in MODEL_TYPES:
        # Run model for each type
        model_results, median_quantity, median_price = run_model_type(
            data, model_type, test_size, price_col, quantity_col, weights_col
        )

        # Check if this model has the lowest mean relative error so far and
        # meets minimum R-squared requirement
        if (
            (model_results[model_type + "_" + best_model_error_col] < best_error)
            &
            # (model_results[model_type + "_r2"] >= min_r2) &
            (model_results[model_type + "_pvalue"] <= max_pvalue)
            & (model_results[model_type + "_" + best_model_error_col] >= 0)
        ):
            best_error = model_results[model_type + "_" + best_model_error_col]
            best_model = model_type

        # Update results with model-specific results
        results.update(model_results)

    # If no best model is found, log and assign NaN values
    if best_model is None:
        logging.info("No best model found")

        # Assign nan
        results["best_model"] = np.nan
        for suffix in MODEL_SUFFIXES_CS:
            results[f"best_{suffix}"] = np.nan
        results["median_quantity"] = np.nan
        results["median_price"] = np.nan

        # Assign False
        for col in ["quality_test", "quality_test_high", "quality_test_medium"]:
            results[col] = False
        results["details"] = np.nan
    else:
        # Assign best model-specific results to the final results using a loop
        results["best_model"] = best_model
        for suffix in MODEL_SUFFIXES_CS:
            results[f"best_{suffix}"] = results[f"{best_model}_{suffix}"]

        # Assign other results
        results["median_quantity"] = median_quantity
        results["median_price"] = median_price

        # Perform quality tests
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

        # Generate detailed results
        results["details"] = make_details(
            results["quality_test"],
            results["quality_test_high"],
            results["power_elasticity"],
        )

    # Convert the dictionary to a DataFrame
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
        logging.info(f"Error for user ID {uid}: {e}")

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
    """Run experiment for multiple user IDs in parallel.

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
    pool = multiprocessing.Pool()  # Use the default number of processes
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
