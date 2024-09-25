"""This module calculate_residuals_with_report."""

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from elasticity.data.configurator import DataColumns


def calculate_residuals_with_report(
    df_by_price: pd.DataFrame, data_columns: DataColumns, p_value_threshold: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate adjusted quantity from a linear regression of quantity on the difference.

    between round price and competitor prices (avg, min, max) for each UID. If the model
    is a good fit (based on at least one p-value below the threshold), return the residuals
    as quantity_adjusted; otherwise, return the original quantity and add a residual_flag column.
    Also, create a report DataFrame for diagnostics.

    Args:
        df_by_price (pd.DataFrame): The unagglomerated data containing quantity, round price,
                                    and competitor prices.
        data_columns (DataColumns): The data columns configuration object containing column names.
        p_value_threshold (float): The p-value threshold to determine if the model is a good fit.
        Defaults to 0.05.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - A modified deep copy of df_by_price with quantity replaced by quantity_adjusted.
            - A report DataFrame with key metrics (R-squared, p-values, coefficients).
    """
    # Make a deep copy of the DataFrame
    df_by_price_copy = df_by_price.copy(deep=True)

    # Initialize the quantity_adjusted and residual flag column
    df_by_price_copy["quantity_adjusted"] = np.nan
    df_by_price_copy["residual_flag"] = 0  # 0: Not a good fit, 1: Good fit

    # Initialize an empty list to collect report data
    report_data = []

    # Loop over each unique UID
    for uid in df_by_price_copy[data_columns.uid].unique():
        # Filter data for the current UID
        uid_data = df_by_price_copy[df_by_price_copy[data_columns.uid] == uid]

        # Calculate the differences between round price and competitor prices (avg, min, max)
        uid_data["diff_avg_competitors_price"] = (
            uid_data[data_columns.round_price] - uid_data[data_columns.avg_competitors_price]
        )
        uid_data["diff_min_competitors_price"] = (
            uid_data[data_columns.round_price] - uid_data[data_columns.min_competitors_price]
        )
        uid_data["diff_max_competitors_price"] = (
            uid_data[data_columns.round_price] - uid_data[data_columns.max_competitors_price]
        )

        # Prepare the data for regression
        x = sm.add_constant(
            uid_data[
                [
                    "diff_avg_competitors_price",
                    "diff_min_competitors_price",
                    "diff_max_competitors_price",
                ]
            ]
        )
        y = uid_data[data_columns.quantity]

        # Fit a linear model: Quantity ~ Price Differences for each UID
        model = sm.OLS(y, x).fit()

        # Check the p-value of the slope coefficients (for avg, min, and max price differences)
        p_values = model.pvalues[
            [
                "diff_avg_competitors_price",
                "diff_min_competitors_price",
                "diff_max_competitors_price",
            ]
        ]

        # Check if at least one p-value is below the threshold
        good_fit = (p_values < p_value_threshold).any()

        # Store the result of the residuals or original quantity
        if good_fit and (model.resid > 0).all():
            # Good fit: Store the residuals in the quantity_adjusted column
            df_by_price_copy.loc[df_by_price_copy[data_columns.uid] == uid, "quantity_adjusted"] = (
                model.resid
            )
            df_by_price_copy.loc[df_by_price_copy[data_columns.uid] == uid, "residual_flag"] = (
                1  # Mark as a good fit
            )
        else:
            # Poor fit: Store the original quantity instead of residuals
            df_by_price_copy.loc[df_by_price_copy[data_columns.uid] == uid, "quantity_adjusted"] = (
                uid_data[data_columns.quantity]
            )

        # Add model details to the report
        report_data.append(
            {
                "uid": uid,
                "r_squared": model.rsquared,
                "p_value_avg_competitors_price": p_values["diff_avg_competitors_price"],
                "p_value_min_competitors_price": p_values["diff_min_competitors_price"],
                "p_value_max_competitors_price": p_values["diff_max_competitors_price"],
                "coef_avg_competitors_price": model.params["diff_avg_competitors_price"],
                "coef_min_competitors_price": model.params["diff_min_competitors_price"],
                "coef_max_competitors_price": model.params["diff_max_competitors_price"],
                "residual_flag": (
                    1 if good_fit else 0
                ),  # Flag indicating if the model was a good fit
            }
        )

    # After processing all UIDs, replace the original quantity column with quantity_adjusted
    df_by_price_copy[data_columns.quantity] = df_by_price_copy["quantity_adjusted"]

    # Convert the report data into a DataFrame
    report_df = pd.DataFrame(report_data)

    # Drop the quantity_adjusted column since we already replaced the quantity column
    df_by_price_copy = df_by_price_copy.drop(columns=["quantity_adjusted"])

    # Calculate and print the number of UIDs with residual_flag = 1
    num_good_fit_uids = df_by_price_copy["residual_flag"].sum()
    print(f"Number of UIDs with residual_flag = 1: {num_good_fit_uids}")

    # Return the modified DataFrame with quantity replaced and the report DataFrame
    return df_by_price_copy, report_df
