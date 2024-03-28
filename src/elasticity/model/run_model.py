"""Module of modeling."""
import logging
import multiprocessing

import numpy as np
import pandas as pd
from elasticity.model.cross_validation import cross_validation
from elasticity.model.model import estimate_coefficients


def run_model_type(data: pd.DataFrame,
                   model_type: str,
                   test_size: float,
                   price_col: str,
                   quantity_col: str,
                   weights_col: str) -> dict:
    """Calculate cross-validation and regression results."""
    results = {}
    median_price = data[price_col].median()
    median_quantity = data[quantity_col].median()
    try:
        # Cross-validation results
        (cv_mean_relative_error,
         cv_mean_a,
         cv_mean_b,
         cv_mean_elasticity,
         cv_mean_r2) = cross_validation(data,
                                        model_type,
                                        test_size,
                                        price_col,
                                        quantity_col,
                                        weights_col)
        # Regular regression results
        a, b, pvalue, r_squared, elasticity = estimate_coefficients(
            data,
            model_type,
            price_col=price_col,
            quantity_col=quantity_col,
            weights_col=weights_col)
        # Store the results in a dictionary
        results[model_type + '_mean_relative_error'] = cv_mean_relative_error
        results[model_type + '_mean_a'] = cv_mean_a
        results[model_type + '_mean_b'] = cv_mean_b
        results[model_type + '_mean_elasticity'] = cv_mean_elasticity
        results[model_type + '_mean_r2'] = cv_mean_r2
        results[model_type + '_a'] = a
        results[model_type + '_b'] = b
        results[model_type + '_pvalue'] = pvalue
        results[model_type + '_r2'] = r_squared
        results[model_type + '_elasticity'] = elasticity
    except Exception as e:
        logging.info("Error in run_model_type: %s", e)
        # Set all the results to np.nan
        results[model_type + '_mean_relative_error'] = np.nan
        results[model_type + '_mean_a'] = np.nan
        results[model_type + '_mean_b'] = np.nan
        results[model_type + '_mean_elasticity'] = np.nan
        results[model_type + '_mean_r2'] = np.nan
        results[model_type + '_a'] = np.nan
        results[model_type + '_b'] = np.nan
        results[model_type + '_pvalue'] = np.nan
        results[model_type + '_r2'] = np.nan
        results[model_type + '_elasticity'] = np.nan
    return results, median_quantity, median_price

def run_experiment(data: pd.DataFrame,
                   test_size: float = 0.1,
                   price_col: str = 'price',
                   quantity_col: str = 'quantity',
                   weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment."""
    results = {}
    best_model = None
    best_error = float('inf')  # Initialize with a very large value
    for model_type in ['linear', 'power', 'exponential']:
        (model_results,
         median_quantity,
         median_price) = run_model_type(data,
                                        model_type,
                                        test_size,
                                        price_col,
                                        quantity_col,
                                        weights_col)

        # Check if this model has the lowest mean relative error so far
        if model_results[model_type + '_mean_relative_error'] < best_error:
            best_error = model_results[model_type + '_mean_relative_error']
            best_model = model_type

        results.update(model_results)

    if best_model is None:
        logging.info("No best model found")
        # Add columns for the best model
        results['best_model'] = np.nan
        results['best_model_a'] = np.nan
        results['best_model_b'] = np.nan
        results['best_model_r2'] = np.nan
        results['best_mean_relative_error'] = np.nan
        results['best_model_elasticity'] = np.nan
        results['median_quantity'] = np.nan
        results['median_price'] = np.nan
    else:
        # Add columns for the best model
        results['best_model'] = best_model
        results['best_model_a'] = results[best_model + '_a']
        results['best_model_b'] = results[best_model + '_b']
        results['best_model_r2'] = results[best_model + '_r2']
        results['best_mean_relative_error'] = results[best_model + '_mean_relative_error']
        results['best_model_elasticity'] = results[best_model + '_elasticity']
        results['median_quantity'] = median_quantity
        results['median_price'] = median_price
        results['quality_test'] = np.where(
            (results['median_quantity'] < 20) &
            (results['best_mean_relative_error'] <= 25), True,
            np.where((results['median_quantity'] >= 20) &
                     (results['median_quantity'] < 100) &
                     (results['best_mean_relative_error'] <= 20), True,
                     np.where((results['median_quantity'] >= 100) &
                              (results['best_mean_relative_error'] <= 15), True,
                              False)))

    # Convert the dictionary to a DataFrame
    return pd.DataFrame(results, index=[0])

def run_experiment_for_uid(uid: str,
                           data: pd.DataFrame,
                           test_size: float = 0.1,
                           price_col: str = 'price',
                           quantity_col: str = 'quantity',
                           weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment for a specific user ID."""
    subset_data = data[data['uid'] == uid]
    try:
        results_df = run_experiment(data=subset_data,
                                    test_size=test_size,
                                    price_col=price_col,
                                    quantity_col=quantity_col,
                                    weights_col=weights_col)
    except Exception as e:
        logging.info("Error for user ID %s: %s", uid, e)
        columns = ['linear_mean_relative_error', 'linear_mean_a', 'linear_mean_b',
           'linear_mean_elasticity', 'linear_mean_r2', 'linear_a', 'linear_b',
           'linear_pvalue', 'linear_r2', 'linear_elasticity', 'power_mean_relative_error',
           'power_mean_a', 'power_mean_b', 'power_mean_elasticity', 'power_mean_r2',
           'power_a', 'power_b', 'power_pvalue', 'power_r2', 'power_elasticity',
           'exponential_mean_relative_error', 'exponential_mean_a', 'exponential_mean_b',
           'exponential_mean_elasticity', 'exponential_mean_r2', 'exponential_a',
           'exponential_b', 'exponential_pvalue', 'exponential_r2', 'exponential_elasticity',
           'best_model', 'best_model_a', 'best_model_b', 'best_model_r2',
           'best_mean_relative_error', 'best_model_elasticity', 'median_quantity',
           'median_price', 'quality_test', 'uid']

        results_df = pd.DataFrame(np.nan, index=[0], columns=columns)
    results_df['uid'] = uid
    return results_df

def run_experiment_for_uids_parallel(df_input: pd.DataFrame,
                                     test_size: float = 0.1,
                                     price_col: str = 'price',
                                     quantity_col: str = 'quantity',
                                     weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment for multiple user IDs in parallel."""
    # Delete rows with price equal to zero
    df_input = df_input[df_input[price_col]!=0]
    unique_uids = df_input['uid'].unique()
    pool = multiprocessing.Pool()  # Use the default number of processes
    results_list = pool.starmap(
        run_experiment_for_uid, [(uid,
                                  df_input,
                                  test_size,
                                  price_col,
                                  quantity_col,
                                  weights_col) for uid in unique_uids])
    pool.close()
    pool.join()
    return pd.concat(results_list)

def run_experiment_for_uids_not_parallel(
        df_input: pd.DataFrame,
        test_size: float = 0.1,
        price_col: str = 'price',
        quantity_col: str = 'quantity',
        weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment for multiple user IDs (not in parallel)."""
    # Delete rows with price equal to zero
    df_input = df_input[df_input[price_col]!=0]
    results_list = []
    unique_uids = df_input['uid'].unique()
    for uid in unique_uids:
        results_df = run_experiment_for_uid(uid,
                                            df_input,
                                            test_size,
                                            price_col,
                                            quantity_col,
                                            weights_col)
        results_list.append(results_df)
    return pd.concat(results_list)
