"""Module of modeling."""

import logging
import multiprocessing

import numpy as np
import pandas as pd

from elasticity.model.cross_validation import cross_validation
from elasticity.model.model import estimate_coefficients


def run_model_type(
    data: pd.DataFrame,
    model_type: str,
    test_size: float,
    price_col: str,
    quantity_col: str,
    weights_col: str,
) -> dict:
    """
    Calculate cross-validation and regression results.

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
        # Cross-validation results
        (
            cv_mean_relative_error,
            cv_mean_a,
            cv_mean_b,
            cv_mean_elasticity,
            cv_mean_r2,
        ) = cross_validation(
            data, model_type, test_size, price_col, quantity_col, weights_col
        )
        # Regular regression results
        a, b, pvalue, r_squared, elasticity, elasticity_error_propagation, aic, relative_absolute_error = (
            estimate_coefficients(
                data,
                model_type,
                price_col=price_col,
                quantity_col=quantity_col,
                weights_col=weights_col,
            )
        )
        # Store the results in a dictionary
        results[model_type + "_mean_relative_error"] = cv_mean_relative_error
        results[model_type + "_mean_a"] = cv_mean_a
        results[model_type + "_mean_b"] = cv_mean_b
        results[model_type + "_mean_elasticity"] = cv_mean_elasticity
        results[model_type + "_mean_r2"] = cv_mean_r2
        results[model_type + "_a"] = a
        results[model_type + "_b"] = b
        results[model_type + "_pvalue"] = pvalue
        results[model_type + "_r2"] = r_squared
        results[model_type + "_elasticity"] = elasticity
        results[model_type + "_relative_absolute_error"] = relative_absolute_error
        results[model_type + "_elasticity_error_propagation"] = (
            elasticity_error_propagation
        )
        results[model_type + "_aic"] = aic
    except Exception as e:
        logging.info(f"Error in run_model_type: {e}")
        # Set all the results to np.nan
        model_type_columns = [
            "_mean_relative_error",
            "_mean_a",
            "_mean_b",
            "_mean_elasticity",
            "_mean_r2",
            "_a",
            "_b",
            "_pvalue",
            "_r2",
            "_elasticity",
            "_relative_absolute_error",
            "_elasticity_error_propagation",
            "_elasticity_error_propagation",
            "_aic"]
        for col in [model_type + c for c in model_type_columns]:
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

    Parameters:
    - median_quantity: median quantity value
    - best_mean_relative_error: mean relative error of the best model
    - high_threshold: if True, apply high threshold, otherwise apply regular threshold
    - q_test_value_1: value for first quality test threshold
    - q_test_value_2: value for second quality test threshold
    - q_test_threshold_1: threshold for the first quality test
    - q_test_threshold_2: threshold for the second quality test
    - q_test_threshold_3: threshold for the third quality test

    Returns:
    - bool: True if the test passes, False otherwise
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

    Parameters:
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


def run_experiment(
    data: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
    min_r2: float = 0.3,
    quality_test_error_col: str = 'best_relative_absolute_error'
) -> pd.DataFrame:
    """Run experiment and return results DataFrame.

    Parameters:
    - data: DataFrame containing the dataset
    - test_size: proportion of the dataset to include in the test split
    - price_col: column name for prices
    - quantity_col: column name for quantities
    - weights_col: column name for weights
    - min_r2: minimum R-squared value required for model acceptance
    - quality_test_error_col: by default best_relative_absolute_error
    other option is best_mean_relative_error
    to take the mean error of the cross validation

    Returns:
    - DataFrame: results of the experiment
    """
    # Initialize variables
    results = {}
    best_model = None
    best_error = float("inf")  # Initialize with a very large value

    # Iterate over different model types
    for model_type in ["linear", "power", "exponential"]:
        # Run model for each type
        model_results, median_quantity, median_price = run_model_type(
            data, model_type, test_size, price_col, quantity_col, weights_col
        )

        # Check if this model has the lowest mean relative error so far and
        # meets minimum R-squared requirement
        if (model_results[model_type + "_mean_relative_error"] < best_error) & (
            model_results[model_type + "_r2"] >= min_r2
        ):
            best_error = model_results[model_type + "_mean_relative_error"]
            best_model = model_type

        # Update results with model-specific results
        results.update(model_results)

    # If no best model is found, log and assign NaN values
    if best_model is None:
        logging.info("No best model found")
        best_model_columns = [
            "best_model",
            "best_model_a",
            "best_model_b",
            "best_model_r2",
            "best_mean_relative_error",
            "best_relative_absolute_error",
            "best_model_elasticity",
            "best_model_elasticity_error_propagation",
            "best_model_aic",
            "median_quantity",
            "median_price",
            "details",
        ]
        for col in best_model_columns:
            results[col] = np.nan

        for col in ["quality_test", "quality_test_high", "quality_test_medium"]:
            results[col] = False
    else:
        # Assign best model-specific results to the final results
        results["best_model"] = best_model
        results["best_model_a"] = results[best_model + "_a"]
        results["best_model_b"] = results[best_model + "_b"]
        results["best_model_r2"] = results[best_model + "_r2"]
        results["best_mean_relative_error"] = results[
            best_model + "_mean_relative_error"
        ]
        results["best_relative_absolute_error"] = results[
            best_model + "_relative_absolute_error"
        ]
        results["best_model_elasticity"] = results[best_model + "_elasticity"]
        results["best_model_elasticity_error_propagation"] = results[
            best_model + "_elasticity_error_propagation"
        ]
        results["best_model_aic"] = results[best_model + "_aic"]
        results["median_quantity"] = median_quantity
        results["median_price"] = median_price

        # Perform quality tests
        results["quality_test"] = quality_test(
            results["median_quantity"], results[quality_test_error_col]
        )
        # Perform high quality test
        results["quality_test_high"] = quality_test(
            results["median_quantity"],
            results[quality_test_error_col],
            high_threshold=True,
        )
        # Perform medium quality test
        results["quality_test_medium"] = (
            results["quality_test"] and not results["quality_test_high"]
        )

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
    """Run experiment for a specific user ID."""
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
        columns = [
            "linear_mean_relative_error",
            "linear_mean_a",
            "linear_mean_b",
            "linear_mean_elasticity",
            "linear_mean_r2",
            "linear_a",
            "linear_b",
            "linear_pvalue",
            "linear_r2",
            "linear_elasticity",
            "linear_elasticity_error_propagation",
            "linear_aic",
            "power_mean_relative_error",
            "power_mean_a",
            "power_mean_b",
            "power_mean_elasticity",
            "power_mean_r2",
            "power_a",
            "power_b",
            "power_pvalue",
            "power_r2",
            "power_elasticity",
            "power_elasticity_error_propagation",
            "power_aic",
            "exponential_mean_relative_error",
            "exponential_mean_a",
            "exponential_mean_b",
            "exponential_mean_elasticity",
            "exponential_mean_r2",
            "exponential_a",
            "exponential_b",
            "exponential_pvalue",
            "exponential_r2",
            "exponential_elasticity",
            "exponential_elasticity_error_propagation",
            "exponential_aic",
            "best_model",
            "best_model_a",
            "best_model_b",
            "best_model_r2",
            "best_mean_relative_error",
            "best_model_elasticity",
            "best_model_elasticity_error_propagation",
            "median_quantity",
            "median_price",
            "quality_test",
            "uid",
        ]

        results_df = pd.DataFrame(np.nan, index=[0], columns=columns)
    results_df["uid"] = uid
    return results_df


def run_experiment_for_uids_parallel(
    df_input: pd.DataFrame,
    test_size: float = 0.1,
    price_col: str = "price",
    quantity_col: str = "quantity",
    weights_col: str = "days",
) -> pd.DataFrame:
    """Run experiment for multiple user IDs in parallel."""
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
    """Run experiment for multiple user IDs (not in parallel)."""
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
