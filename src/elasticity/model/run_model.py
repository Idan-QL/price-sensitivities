"""Module of modeling."""
import numpy as np
import pandas as pd
import multiprocessing
from elasticity.model.cross_validation import cross_validation
from elasticity.model.model import estimate_coefficients

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
        a, b, pvalue, r_squared, elasticity, median_quantity, median_price = estimate_coefficients(
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

        # Check if this model has the lowest mean relative error so far
        if cv_mean_relative_error < best_error:
            best_error = cv_mean_relative_error
            best_model = model_type
    
    if best_model is None:
        print("No best model found")
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
    results_df = pd.DataFrame(results, index=[0])

    return results_df

def run_experiment_for_uid(uid: str,
                           data: pd.DataFrame,
                           test_size: float = 0.1,
                           price_col: str = 'price',
                           quantity_col: str = 'quantity',
                           weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment for a specific user ID."""
    subset_data = data[data['uid'] == uid]
    results_df = run_experiment(data=subset_data,
                                test_size=test_size,
                                price_col=price_col,
                                quantity_col=quantity_col,
                                weights_col=weights_col)
    results_df['uid'] = uid
    return results_df

def run_experiment_for_uids_parallel(df: pd.DataFrame,
                                     test_size: float = 0.1,
                                     price_col: str = 'price',
                                     quantity_col: str = 'quantity',
                                     weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment for multiple user IDs in parallel."""
    unique_uids = df['uid'].unique()
    pool = multiprocessing.Pool()  # Use the default number of processes
    results_list = pool.starmap(
        run_experiment_for_uid, [(uid,
                                  df,
                                  test_size,
                                  price_col,
                                  quantity_col,
                                  weights_col) for uid in unique_uids])
    pool.close()
    pool.join()
    return pd.concat(results_list)

def run_experiment_for_uids_not_parallel(
        df: pd.DataFrame,
        test_size: float = 0.1,
        price_col: str = 'price',
        quantity_col: str = 'quantity',
        weights_col: str = 'days') -> pd.DataFrame:
    """Run experiment for multiple user IDs (not in parallel)."""
    results_list = []
    unique_uids = df['uid'].unique()
    for uid in unique_uids:
        results_df = run_experiment_for_uid(uid, df, test_size, price_col, quantity_col, weights_col)
        results_list.append(results_df)
    return pd.concat(results_list)
