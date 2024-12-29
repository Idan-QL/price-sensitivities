"""Module of utils_sensitivity."""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from elasticity.data.configurator import DataColumns, SensitivityParameters


def run_granger_test(
    series_1: pd.Series,
    series_2: pd.Series,
    max_lag: int,
) -> Optional[Dict[str, Any]]:
    """Run Granger causality test between two series.

    Args:
        series_1 (pd.Series): The time series hypothesized as the effect.
        series_2 (pd.Series): The time series hypothesized as the cause.
        max_lag (int): The maximum number of lags to test for Granger causality.

    Returns:
        Optional[Dict[str, Any]]
        Dictionary with p-values and causality flag, or None if the test fails.
    """
    try:
        data = pd.concat([series_1, series_2], axis=1)
        data.columns = ["effect", "cause"]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=FutureWarning, message=".*verbose is deprecated.*"
            )
            granger_test = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        p_values = [round(granger_test[lag][0]["ssr_ftest"][1], 4) for lag in range(1, max_lag + 1)]
        causality = any(p < 0.05 for p in p_values)
        return {"p_values": p_values, "causality": causality}

    except Exception as e:
        logging.error(f"Error during Granger test: {e}")
        return None


def granger_causality_test(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 5,
    max_diff: int = 2,
) -> Optional[Dict[str, Any]]:
    """Perform a Granger causality test in both directions.

    Handles constant series, non-stationary data, insufficient sample sizes,
    and missing values gracefully.

    Args:
        cause (pd.Series): The time series hypothesized to cause the effect.
        effect (pd.Series): The time series hypothesized to be affected by the cause.
        max_lag (int, optional): The maximum number of lags to test for Granger causality.
        Default is 5.
        max_diff (int, optional): The maximum number of differences to apply to achieve
        stationarity. Default is 2.

    Returns:
        Optional[Dict[str, Any]]
        Dictionary with p-values and causality flags for both directions, or None if the test
        cannot be performed.

    Example:
        {
            'p_values_cause_to_effect': [0.0033, 0.0066, 0.0087],
            'causality_cause_to_effect': True,
            'p_values_effect_to_cause': [0.0456, 0.0789, 0.1234],
            'causality_effect_to_cause': False
        }
    """
    try:
        # Drop NaN values
        combined_df = pd.concat([cause, effect], axis=1).dropna()
        cause = combined_df.iloc[:, 0]
        effect = combined_df.iloc[:, 1]

        # Ensure sufficient data for Granger causality
        if len(cause) <= max_lag + 1:
            # logging.warning(f"Insufficient data for Granger causality: {len(cause)} rows.")
            return None

        # Ensure variance in both series
        if cause.std() == 0 or effect.std() == 0:
            # logging.warning("One or both series have zero variance. Skipping test.")
            return None

        # Make both series stationary
        cause_stationary = make_stationary(cause, max_diff=max_diff)
        effect_stationary = make_stationary(effect, max_diff=max_diff)

        if cause_stationary is None or effect_stationary is None:
            # logging.warning("Stationarity could not be achieved for one or both series.")
            return None

        # Align after differencing
        min_length = min(len(cause_stationary), len(effect_stationary))
        cause_stationary = cause_stationary[-min_length:]
        effect_stationary = effect_stationary[-min_length:]

        # Run Granger tests in both directions
        result_cause_to_effect = run_granger_test(
            series_1=cause_stationary, series_2=effect_stationary, max_lag=max_lag
        )
        result_effect_to_cause = run_granger_test(
            series_1=effect_stationary, series_2=cause_stationary, max_lag=max_lag
        )

        return {
            "p_values_cause_to_effect": (
                result_cause_to_effect.get("p_values") if result_cause_to_effect else None
            ),
            "causality_cause_to_effect": (
                result_cause_to_effect.get("causality") if result_cause_to_effect else None
            ),
            "p_values_effect_to_cause": (
                result_effect_to_cause.get("p_values") if result_effect_to_cause else None
            ),
            "causality_effect_to_cause": (
                result_effect_to_cause.get("causality") if result_effect_to_cause else None
            ),
        }

    except Exception as e:
        logging.error(f"Unexpected error during Granger causality test: {e}")
        return None


def compute_corr(sub_df: pd.DataFrame, col1: str, col2: str) -> float:
    """Compute the Pearson correlation between two columns in a dataframe subset.

    Args:
        sub_df (pd.DataFrame): Subset of the main DataFrame for a specific competitor.
        col1 (str): Name of the first column.
        col2 (str): Name of the second column.

    Returns:
        float
        Pearson correlation coefficient or 0 if not computable.
    """
    # Check if columns exist
    if col1 not in sub_df.columns or col2 not in sub_df.columns:
        return np.nan

    # Drop rows with NaN or infinite values
    valid = sub_df[[col1, col2]].replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure sufficient non-NaN data
    if len(valid) < 2:
        return 0

    # Check for zero variance in either column
    if valid[col1].std() == 0 or valid[col2].std() == 0:
        return 0

    # Compute and return the Pearson correlation
    return valid[col1].corr(valid[col2])


def make_stationary(series: pd.Series, max_diff: int = 2) -> Optional[pd.Series]:
    """Make a time series stationary by differencing up to max_diff times.

    Handles constant series and insufficient sample sizes by returning None.

    Args:
        series (pd.Series): The time series data.
        max_diff (int): Maximum number of differences to apply.

    Returns:
        pd.Series or None
        Stationary series if achieved, else None.
    """
    # Check if the series is constant
    if series.min() == series.max():
        logging.warning("Series is constant. Skipping stationarity transformation.")
        return None

    try:
        for _ in range(max_diff + 1):
            result = adfuller(series.dropna())
            p_value = result[1]
            if p_value < 0.05:
                return series
            series = series.diff().dropna()
            if series.min() == series.max():
                logging.warning(
                    "Series became constant after differencing. Cannot achieve stationarity."
                )
                return None
    except ValueError as ve:
        logging.error(f"ADF test failed: {ve}. Skipping stationarity transformation.")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during stationarity transformation: {e}. Skipping.")
        return None

    return None


def calculate_coverage(df: pd.DataFrame, date_column: str) -> float:
    """Calculate the percentage of days covered in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to evaluate.
        date_column (str): The name of the date column.

    Returns:
        float:
            The percentage of days covered based on the range between the minimum and maximum dates.
    """
    # Check if the column is already datetime, otherwise convert
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Calculate the range of days
    min_date = df[date_column].min()
    max_date = df[date_column].max()

    # Handle edge case where min_date or max_date is NaT
    if pd.isna(min_date) or pd.isna(max_date):
        return None

    # Total days in the range
    total_days = (max_date - min_date).days + 1

    # Unique days in the column
    num_unique_days = df[date_column].nunique()

    # Calculate coverage percentage
    return round((num_unique_days / total_days) * 100, 2)


def initialize_result_for_insufficient_coverage(
    correlation_pairs: List[Tuple[str, str]], granger_pairs: List[Tuple[str, str]]
) -> Dict[str, Optional[float]]:
    """Initialize result dictionary with NaN for insufficient coverage.

    Args:
        correlation_pairs (list of tuples): List of column pairs for correlation.
        granger_pairs (list of tuples): List of column pairs for Granger causality.

    Returns:
        dict
        A dictionary with NaN values for all correlation and Granger fields.
    """
    result = {}

    for col1, col2 in correlation_pairs:
        result[f"corr_{col1}_{col2}_overall"] = np.nan
        result[f"corr_{col1}_{col2}_quantity_gt0"] = np.nan

    for col1, col2 in granger_pairs:
        set_granger_results_to_nan(result, col1, col2)

    return result


def calculate_correlations(
    df: pd.DataFrame,
    correlation_pairs: List[Tuple[str, str]],
    quantity_column: str,
    min_abs_correlation: float,
) -> Dict[str, Optional[float]]:
    """Calculate correlations for the given pairs of columns.

    Args:
        df (pd.DataFrame): The DataFrame to evaluate.
        correlation_pairs (list of tuples): List of column pairs for correlation.
        quantity_column (str): The column to use for filtering rows (e.g., where quantity > 0).
        min_abs_correlation (float): Minimum absolute correlation threshold for the
        correlation test to be considered `True`.

    Returns:
        dict
        A dictionary containing correlation scores and boolean results for each pair.
    """

    def compute_and_check_correlation(
        df_subset: pd.DataFrame, col1: str, col2: str
    ) -> Tuple[Optional[float], bool]:
        """Compute correlation and check against the threshold."""
        if df_subset.empty:
            return np.nan, False
        corr = compute_corr(df_subset, col1, col2)
        return corr, abs(corr) >= min_abs_correlation

    result = {}

    for col1, col2 in correlation_pairs:
        # Skip if required columns are missing
        if col1 not in df.columns or col2 not in df.columns:
            result.update(
                {
                    f"score_corr_{col1}_{col2}_overall": np.nan,
                    f"corr_{col1}_{col2}_overall": False,
                    f"score_corr_{col1}_{col2}_quantity_gt0": np.nan,
                    f"corr_{col1}_{col2}_quantity_gt0": False,
                }
            )
            continue

        # Overall correlation
        overall_corr, overall_flag = compute_and_check_correlation(df, col1, col2)
        result[f"score_corr_{col1}_{col2}_overall"] = overall_corr
        result[f"corr_{col1}_{col2}_overall"] = overall_flag

        # Conditional correlation (quantity > 0)
        if quantity_column in df.columns:
            conditional_corr, conditional_flag = compute_and_check_correlation(
                df[df[quantity_column] > 0], col1, col2
            )
            result[f"score_corr_{col1}_{col2}_quantity_gt0"] = conditional_corr
            result[f"corr_{col1}_{col2}_quantity_gt0"] = conditional_flag
        else:
            result.update(
                {
                    f"score_corr_{col1}_{col2}_quantity_gt0": np.nan,
                    f"corr_{col1}_{col2}_quantity_gt0": False,
                }
            )

    return result


def set_granger_results_to_nan(result: Dict[str, Optional[float]], col1: str, col2: str) -> None:
    """Helper function to set Granger causality results to NaN for a specific column pair.

    Args:
        result (dict): The dictionary to update.
        col1 (str): The first column name in the pair.
        col2 (str): The second column name in the pair.
    """
    result[f"granger_p_values_{col1}_to_{col2}"] = np.nan
    result[f"granger_cause_{col1}_to_{col2}"] = np.nan
    result[f"granger_p_values_{col2}_to_{col1}"] = np.nan
    result[f"granger_cause_{col2}_to_{col1}"] = np.nan


def calculate_granger_causality(
    df: pd.DataFrame, granger_pairs: List[Tuple[str, str]], max_lag: int, max_diff: int
) -> Dict[str, Optional[float]]:
    """Perform Granger causality tests for the given pairs of columns.

    Args:
        df (pd.DataFrame): The DataFrame to evaluate.
        granger_pairs (list of tuples): List of column pairs for Granger causality.
        max_lag (int): Maximum number of lags for Granger causality.
        max_diff (int): Maximum number of differences to apply for stationarity.

    Returns:
        dict
        A dictionary containing Granger causality results for both directions for each pair.
    """
    result = {}

    for col1, col2 in granger_pairs:
        if col1 in df.columns and col2 in df.columns:
            # Perform Granger causality tests in both directions
            causality_results = granger_causality_test(
                cause=df[col1], effect=df[col2], max_lag=max_lag, max_diff=max_diff
            )

            if causality_results is not None:
                # Cause -> Effect
                result[f"granger_p_values_{col1}_to_{col2}"] = causality_results.get(
                    "p_values_cause_to_effect", np.nan
                )
                result[f"granger_cause_{col1}_to_{col2}"] = causality_results.get(
                    "causality_cause_to_effect", np.nan
                )
                # Effect -> Cause
                result[f"granger_p_values_{col2}_to_{col1}"] = causality_results.get(
                    "p_values_effect_to_cause", np.nan
                )
                result[f"granger_cause_{col2}_to_{col1}"] = causality_results.get(
                    "causality_effect_to_cause", np.nan
                )
            else:
                set_granger_results_to_nan(result, col1, col2)
        else:
            set_granger_results_to_nan(result, col1, col2)

    return result


def correlation_test(
    df_uid: pd.DataFrame,
    correlation_pairs: List[Tuple[str, str]],
    granger_pairs: List[Tuple[str, str]],
    data_columns: DataColumns,
    sensitivity_parameters: Optional[SensitivityParameters],
) -> pd.Series:
    """Perform correlation and Granger causality tests on a unique DataFrame subset.

    Args:
        df_uid (pd.DataFrame): The subset DataFrame for a specific group.
        correlation_pairs (list of tuples): List of column pairs for correlation tests.
        granger_pairs (list of tuples): List of column pairs for Granger causality tests.
        data_columns (DataColumns): Object containing column names for the analysis.
        sensitivity_parameters : SensitivityParameters.
        Default SensitivityParameters()

    Returns:
        pd.Series
        A Series containing correlation coefficients, % days covered, Granger causality results,
        a `correlation_test` column that is True if any test is True, and a `details` column
        listing all the True tests.
    """
    result = {}

    # Calculate % days covered
    percent_days_covered = calculate_coverage(df_uid, data_columns.date)
    result["%_days_covered"] = percent_days_covered

    sensitivity_parameters = sensitivity_parameters or SensitivityParameters()
    coverage_threshold = sensitivity_parameters.coverage_threshold
    min_abs_correlation = sensitivity_parameters.min_abs_correlation
    max_lag = sensitivity_parameters.max_lag
    max_diff = sensitivity_parameters.max_diff

    # Handle insufficient coverage
    if percent_days_covered is None or percent_days_covered < coverage_threshold:
        insufficient_result = initialize_result_for_insufficient_coverage(
            correlation_pairs, granger_pairs
        )
        result.update(insufficient_result)
        result["correlation_test"] = False
        result["details_correlation_test"] = "low coverage"
        return pd.Series(result)

    # Calculate correlations
    correlation_result = calculate_correlations(
        df=df_uid,
        correlation_pairs=correlation_pairs,
        quantity_column=data_columns.quantity,
        min_abs_correlation=min_abs_correlation,
    )
    result.update(correlation_result)

    # Identify correlation tests exceeding the minimum absolute correlation threshold
    correlation_tests = [
        key for key, value in result.items() if key.startswith("corr_") and bool(value) is True
    ]

    # Perform Granger causality tests
    granger_result = calculate_granger_causality(df_uid, granger_pairs, max_lag, max_diff)
    result.update(granger_result)

    # Identify all tests that are True
    granger_tests = [
        key
        for key, value in result.items()
        if key.startswith("granger_cause_") and bool(value) is True
    ]

    # Combine correlation and Granger test results
    all_true_tests = correlation_tests + granger_tests

    # Set 'test' and 'details'
    if all_true_tests:
        result["correlation_test"] = True
        result["details_correlation_test"] = ", ".join(all_true_tests)
    else:
        result["correlation_test"] = False
        result["details_correlation_test"] = "All test False"

    return pd.Series(result)


def run_correlation_test(
    raw_df: pd.DataFrame,
    data_columns: DataColumns,
    sensitivity_parameters: SensitivityParameters,
) -> pd.DataFrame:
    """Run correlation tests with default parameters on the provided DataFrame.

    Args:
        raw_df (pd.DataFrame): The raw DataFrame containing the data to analyze.
        data_columns (DataColumns): Configuration for column mappings.
        sensitivity_parameters (SensitivityParameters): Sensitivity parameters.

    Returns:
        pd.DataFrame: A summary DataFrame containing the results of the analysis.
    """
    # Convert 'date' column to datetime
    raw_df[data_columns.date] = pd.to_datetime(raw_df[data_columns.date], errors="coerce")

    # Define correlation and Granger pairs
    correlation_pairs = [(data_columns.ratio_shelf_price_competitor, data_columns.quantity)]
    granger_pairs = [(data_columns.shelf_price, data_columns.competitor_price)]

    # Group the DataFrame by 'uid' and apply the test function
    grouped = raw_df.groupby(data_columns.uid)
    results = grouped.apply(
        lambda group: correlation_test(
            df_uid=group,
            correlation_pairs=correlation_pairs,
            granger_pairs=granger_pairs,
            data_columns=data_columns,
            sensitivity_parameters=sensitivity_parameters,
        )
    )
    # print('type(results)', type(results))
    # type(results) <class 'pandas.core.series.Series'>
    # results.to_pickle("results.pkl")

    # ERROR:root:Unexpected error: agg function failed [how->mean,dtype->object]
    # return results_df.pivot_table(
    #     index=results_df.columns[0],
    #     columns=results_df.columns[1],
    #     values=results_df.columns[2]).reset_index()
    return results.unstack().reset_index()
