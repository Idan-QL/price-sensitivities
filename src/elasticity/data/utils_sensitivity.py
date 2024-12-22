"""Module of utils_sensitivity."""

import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def perform_bi_directional_granger_tests(
    df: pd.DataFrame, max_lag: int = 5, max_diff: int = 2
) -> dict:
    """Perform Granger causality tests.

    In both directions between 'competitor_price' and 'shelf_price_base'.
    Handles constant series and insufficient sample sizes by returning appropriate flags.

    Parameters:
    ----------
    df : pd.DataFrame
        DataFrame containing 'competitor_price' and 'shelf_price_base' columns.
    max_lag : int, optional
        The maximum number of lags to test for Granger causality. Default is 5.
    max_diff : int, optional
        The maximum number of differences to apply to achieve stationarity. Default is 2.

    Returns:
    -------
    dict
        Dictionary containing p-values and causality flags for both directions and
        the merged 'Granger_Causes'.

    Example:
            {
                'p_values_competitor_to_shelf': [...],
                'causality_competitor_to_shelf': True,
                'p_values_shelf_to_competitor': [...],
                'causality_shelf_to_competitor': False,
                'Granger_Causes': True
            }
    """
    group_identifier = (
        df["uid_competitor_name"].iloc[0] if "uid_competitor_name" in df.columns else "unknown"
    )

    causality_result_1 = granger_causality_test(
        cause=df["competitor_price"],
        effect=df["shelf_price_base"],
        max_lag=max_lag,
        max_diff=max_diff,
        group_identifier=group_identifier,
    )

    causality_result_2 = granger_causality_test(
        cause=df["shelf_price_base"],
        effect=df["competitor_price"],
        max_lag=max_lag,
        max_diff=max_diff,
        group_identifier=group_identifier,
    )

    result = {}

    if causality_result_1:
        result["p_values_competitor_to_shelf"] = causality_result_1["p_values"]
        result["causality_competitor_to_shelf"] = causality_result_1["causality"]
    else:
        result["p_values_competitor_to_shelf"] = None
        result["causality_competitor_to_shelf"] = False

    if causality_result_2:
        result["p_values_shelf_to_competitor"] = causality_result_2["p_values"]
        result["causality_shelf_to_competitor"] = causality_result_2["causality"]
    else:
        result["p_values_shelf_to_competitor"] = None
        result["causality_shelf_to_competitor"] = False

    causality_flags = [
        causality_result_1["causality"] if causality_result_1 else False,
        causality_result_2["causality"] if causality_result_2 else False,
    ]

    result["Granger_Causes"] = any(causality_flags)

    return result


def compute_corr(sub_df: pd.DataFrame, col1: str, col2: str) -> float:
    """Compute the Pearson correlation between two columns in a dataframe subset.

    Parameters:
    ----------
    sub_df : pd.DataFrame
        Subset of the main DataFrame for a specific competitor.
    col1 : str
        Name of the first column.
    col2 : str
        Name of the second column.

    Returns:
    -------
    float
        Pearson correlation coefficient or np.nan if not computable.
    """
    valid = sub_df[[col1, col2]].dropna()
    if len(valid) < 2:
        return np.nan
    return valid[col1].corr(valid[col2])


def make_stationary(series: pd.Series, max_diff: int = 2) -> Optional[pd.Series]:
    """Make a time series stationary by differencing up to max_diff times.

    Handles constant series and insufficient sample sizes by returning None.

    Parameters
    ----------
    series : pd.Series
        The time series data.
    max_diff : int
        Maximum number of differences to apply.

    Returns:
    -------
    pd.Series or None
        Stationary series if achieved, else None.
    """
    # Check if the series is constant
    if series.min() == series.max():
        logging.warning("Series is constant. Skipping stationarity transformation.")
        return None

    try:
        for d in range(max_diff + 1):
            result = adfuller(series.dropna())
            p_value = result[1]
            if p_value < 0.05:
                logging.info(
                    f"Series is stationary with p-value={p_value} at differencing level {d}."
                )
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


def granger_causality_test(
    cause: pd.Series,
    effect: pd.Series,
    max_lag: int = 5,
    max_diff: int = 2,
    group_identifier: str = "unknown",
) -> Optional[Dict[str, Any]]:
    """Perform Granger causality test to determine if 'cause' Granger-causes 'effect'.

    Handles constant series and insufficient sample sizes by returning None.

    Parameters
    ----------
    cause : pd.Series
        The time series hypothesized to cause the effect.
    effect : pd.Series
        The time series hypothesized to be affected by the cause.
    max_lag : int, optional
        The maximum number of lags to test for Granger causality. Default is 5.
    max_diff : int, optional
        The maximum number of differences to apply to achieve stationarity. Default is 2.
    group_identifier : str, optional
        Identifier for the group being tested (used for logging). Default is 'unknown'.

    Returns:
    -------
    dict or None
        If successful, returns a dictionary with p-values and causality flag.

    Example:
    -------
        {
            'p_values': [0.0033, 0.0066, 0.0087, 0.0123, 0.0150],
            'causality': True
        }
        If stationarity cannot be achieved or sample size is insufficient, returns None.
    """
    # Ensure both series are aligned
    combined_df = pd.concat([cause, effect], axis=1).dropna()
    cause_aligned = combined_df.iloc[:, 0]
    effect_aligned = combined_df.iloc[:, 1]

    # Make both series stationary
    cause_stationary = make_stationary(cause_aligned, max_diff=max_diff)
    effect_stationary = make_stationary(effect_aligned, max_diff=max_diff)

    if cause_stationary is None or effect_stationary is None:
        logging.warning(
            f"Granger causality test skipped for group '{group_identifier}' due to constant "
            "series or insufficient data."
        )
        return None

    # Align the series after differencing
    min_length = min(len(cause_stationary), len(effect_stationary))
    cause_stationary = cause_stationary[-min_length:]
    effect_stationary = effect_stationary[-min_length:]

    # Define minimum required observations
    min_required_obs = 10 * max_lag + 1  # Example: 10 times max_lag + 1

    if min_length < min_required_obs:
        logging.warning(
            f"Granger causality test skipped for group '{group_identifier}' due to insufficient "
            f"sample size ({min_length} observations). Required: {min_required_obs}."
        )
        return None

    # Combine into DataFrame
    data_granger = pd.concat([effect_stationary, cause_stationary], axis=1)
    data_granger.columns = ["effect", "cause"]

    try:
        with warnings.catch_warnings():
            # Suppress only the specific FutureWarning related to 'verbose'
            warnings.filterwarnings(
                "ignore", category=FutureWarning, message=".*verbose is deprecated.*"
            )
            # Perform Granger Causality Test without verbose
            granger_test = grangercausalitytests(data_granger, maxlag=max_lag, verbose=False)

        # Collect p-values for each lag
        p_values = []
        for lag in range(1, max_lag + 1):
            p_value = granger_test[lag][0]["ssr_ftest"][1]
            p_values.append(round(p_value, 4))

        # Determine causality: any p-value < 0.05
        causality = any(p < 0.05 for p in p_values)

        logging.info(
            f"Granger causality test completed for group '{group_identifier}'. "
            f"Causality: {causality}."
        )
        return {"p_values": p_values, "causality": causality}
    except ValueError as ve:
        logging.error(f"Granger causality test failed for group '{group_identifier}': {ve}")
        return None
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during Granger causality test for group "
            f"'{group_identifier}': {e}"
        )
        return None


# def test(
#     df_group,
#     correlation_pairs=[
#         ("ratio_shelf_price_competitor", "quantity"),
#         ("diff_shelf_price_minus_competitor_price", "quantity"),
#         ("shelf_price_base", "competitor_price"),
#     ],
#     max_lag=5,
#     max_diff=2,
#     total_days=None,
#     coverage_threshold=30,
# ):
#     """Perform correlation and Granger causality tests on a unique DataFrame subset.
#     If the percentage of days covered is below the specified threshold or is None,
#       set all results to NaN.

#     Parameters:
#     ----------
#     df_group : pd.DataFrame
#         The unique DataFrame subset for a specific 'uid_competitor_name'.
#     correlation_pairs : list of tuples, optional
#         List of column pairs to compute Pearson correlation. Default is predefined pairs.
#     max_lag : int, optional
#         The maximum number of lags to test for Granger causality. Default is 5.
#     max_diff : int, optional
#         The maximum number of differences to apply to achieve stationarity. Default is 2.
#     total_days : int, optional
#         Total number of days in the analysis period for percentage calculation. Default is None.
#     coverage_threshold : float, optional
#         The minimum percentage of days required to perform tests. Default is 30.

#     Returns:
#     -------
#     pd.Series
#         A Series containing correlation coefficients, % days covered, Granger causality results,
#         or NaNs if coverage is below the threshold or is None.
#     """
#     result = {}

#     # Extract 'uid_competitor_name' from the group
#     if "uid_competitor_name" in df_group.columns:
#         unique_identifier = df_group["uid_competitor_name"].iloc[0]
#         result["uid_competitor_name"] = unique_identifier
#     else:
#         unique_identifier = "unknown"
#         result["uid_competitor_name"] = unique_identifier

#     # Calculate percentage of days covered
#     if total_days and "date" in df_group.columns:
#         num_unique_days = df_group["date"].nunique()
#         percent_days_covered = (num_unique_days / total_days) * 100
#         percent_days_covered = round(percent_days_covered, 2)
#         result["%_days_covered"] = percent_days_covered
#     else:
#         percent_days_covered = None
#         result["%_days_covered"] = np.nan

#     # Check if coverage meets the threshold or is None
#     if percent_days_covered is None or percent_days_covered < coverage_threshold:
#         logging.info(
#             f"Group '{unique_identifier}' has {percent_days_covered}% days covered
# (< {coverage_threshold}%)
# or coverage is None. Setting all results to NaN."
#         )
#         # Define all expected result fields
#         expected_fields = [
#             "corr_ratio_shelf_price_competitor_quantity_overall",
#             "corr_ratio_shelf_price_competitor_quantity_total_units_sold_gt0",
#             "corr_diff_shelf_price_minus_competitor_price_quantity_overall",
#             "corr_diff_shelf_price_minus_competitor_price_quantity_total_units_sold_gt0",
#             "corr_shelf_price_base_competitor_price_overall",
#             "corr_shelf_price_base_competitor_price_total_units_sold_gt0",
#             "Granger_Causality_p_values_competitor_price_to_shelf_price_base",
#             "Granger_Causes_competitor_price_to_shelf_price_base",
#             "Granger_Causality_p_values_shelf_price_base_to_competitor_price",
#             "Granger_Causes_shelf_price_base_to_competitor_price",
#             "Granger_Causes",
#         ]
#         # Set all fields to NaN
#         for field in expected_fields:
#             result[field] = np.nan
#         return pd.Series(result)

#     # Proceed with calculations if coverage is sufficient
#     # Calculate correlations
#     for col1, col2 in correlation_pairs:
#         # Check if both columns exist
#         if col1 not in df_group.columns or col2 not in df_group.columns:
#             result[f"corr_{col1}_{col2}_overall"] = np.nan
#             result[f"corr_{col1}_{col2}_quantity_gt0"] = np.nan
#             continue

#         # Overall correlation
#         corr_overall = compute_corr(df_group, col1, col2)
#         result[f"corr_{col1}_{col2}_overall"] = corr_overall

#         # Conditional correlation (quantity > 0)
#         if "quantity" in df_group.columns:
#             cond_group = df_group[df_group["quantity"] > 0]
#             corr_conditional = compute_corr(cond_group, col1, col2)
#             result[f"corr_{col1}_{col2}_quantity_gt0"] = corr_conditional
#         else:
#             result[f"corr_{col1}_{col2}_tquantity_gt0"] = np.nan

#     # Perform Granger Causality Tests in Both Directions
#     if "competitor_price" in df_group.columns and "shelf_price_base" in df_group.columns:
#         causality_results = perform_bi_directional_granger_tests(
#             df=df_group, max_lag=max_lag, max_diff=max_diff
#         )

#         # Populate Granger Causality Results
#         result["Granger_Causality_p_values_competitor_price_to_shelf_price_base"] = (
#             causality_results.get("p_values_competitor_to_shelf", None)
#         )
#         result["Granger_Causes_competitor_price_to_shelf_price_base"] = causality_results.get(
#             "causality_competitor_to_shelf", False
#         )
#         result["Granger_Causality_p_values_shelf_price_base_to_competitor_price"] = (
#             causality_results.get("p_values_shelf_to_competitor", None)
#         )
#         result["Granger_Causes_shelf_price_base_to_competitor_price"] = causality_results.get(
#             "causality_shelf_to_competitor", False
#         )
#         result["Granger_Causes"] = causality_results.get("Granger_Causes", False)
#     else:
#         # If necessary columns are missing, set Granger causality results to None or False
#         result["Granger_Causality_p_values_competitor_price_to_shelf_price_base"] = None
#         result["Granger_Causes_competitor_price_to_shelf_price_base"] = False
#         result["Granger_Causality_p_values_shelf_price_base_to_competitor_price"] = None
#         result["Granger_Causes_shelf_price_base_to_competitor_price"] = False
#         result["Granger_Causes"] = False

#     return pd.Series(result)
