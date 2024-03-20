"""Module of modeling."""
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import multiprocessing

def calculate_quantity_from_price(price: float, a: float, b: float, model_type: str) -> float:
    """Calculate quantity demanded given price and demand model parameters.

    Args:
        price (float): Price of the product.
        a (float): Intercept parameter of the demand model.
        b (float): Slope parameter of the demand model.
        model_type (str): Type of demand model ('linear', 'power', or 'exponential').

    Returns:
        float: Quantity demanded.
    """
    if model_type == 'linear':
        return linear_demand(price, a, b)
    elif model_type == 'power':
        return power_demand(price, a, b)
    elif model_type == 'exponential':
        return exponential_demand(price, a, b)
    else:
        raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")

def linear_demand(price: float, a: float, b: float) -> float:
    """Linear demand model: Q = a + bP"""
    return a + b * price

def power_demand(price: float, a: float, b: float) -> float:
    """Power demand model also called constant elasticity: Q = a * P**b"""
    log_price = np.log(price)
    log_q = linear_demand(log_price, a, b) # return log Q
    return np.exp(log_q)

def exponential_demand(price: float, a: float, b: float) -> float:
    """Exponential demand model: Q = a * exp(-bP)"""
    log_q = linear_demand(price, a, b)
    return np.exp(log_q)

def linear_elasticity(price: float, a: float, b: float) -> float:
    """Linear demand model elasticity: e = -bP / (a + bP)"""
    demand = linear_demand(price, a, b)
    if demand == 0:
        return float('inf') if price == 0 else float('-inf')
    return b * price / demand

def power_elasticity(b: float) -> float:
    """Power demand model elasticity: e = b"""
    return b

def exponential_elasticity(price: float, b: float) -> float:
    """Exponential demand model elasticity: e = b * price"""
    return b * price

def calculate_elasticity_from_parameters(model_type: str, a: float, b: float, price: float) -> float:
    """Calculate price elasticity of demand (PED) given coefficients and price points."""
    if model_type == 'linear' :
        elasticity = linear_elasticity(price, a, b)
    elif model_type == 'power':
        elasticity = power_elasticity(b)
    elif model_type == 'exponential':
        elasticity = exponential_elasticity(price, b)
    else:
        raise ValueError("Invalid model type. Use 'linear', 'power', or 'exponential'.")
    return round(elasticity, 2)

def estimate_coefficients(data: pd.DataFrame,
                          model_type: str,
                          price_col: str = 'price',
                          quantity_col: str = 'quantity',
                          weights_col: str = 'days') -> Tuple[float, float, float, float, float]:
    """Estimate coefficients for demand model using log transformation if nonlinear."""
    X = sm.add_constant(data[[price_col]])
    y = data[quantity_col]
    weights = data[weights_col]
    
    if model_type == 'power':
        y = np.log(y)
        X[price_col] = np.log(X[price_col])
    elif model_type == 'exponential':
        y = np.log(y)
    
    model = sm.WLS(y, X, weights=weights).fit()
    pvalue = model.f_pvalue
    r_squared = model.rsquared

    a, b = model.params.iloc[0], model.params.iloc[1]
    price_point = data[price_col].median()
    elasticity = calculate_elasticity_from_parameters(model_type, a, b, price_point)
    return a, b, pvalue, r_squared, elasticity

def cross_validation(data: pd.DataFrame,
                     model_type: str,
                     test_size: float = 0.1,
                     price_col: str = 'price',
                     quantity_col: str = 'quantity',
                     weights_col: str = 'days',
                     n_tests: int = 3) -> Tuple[float, float, float, float, float]:
    """Perform cross-validation."""
    relative_errors = []
    a_lists = []
    b_lists = []
    elasticity_lists = []
    r_squared_lists = []
    for i in range(n_tests):
        data_train, data_test = train_test_split(data, test_size=test_size, random_state=42 + i)
        a, b, pvalue, r_squared, elasticity = estimate_coefficients(data_train,
                                                                    model_type,
                                                                    price_col=price_col,
                                                                    quantity_col=quantity_col,
                                                                    weights_col=weights_col)
        predicted_quantity = [calculate_quantity_from_price(p, a, b, model_type) for p in data_test[price_col]]
        absolute_errors = np.abs(data_test[quantity_col] - predicted_quantity)
        relative_error = np.mean(absolute_errors / data_test[quantity_col]) * 100  # Calculate as a percentage
        relative_errors.append(relative_error)
        a_lists.append(a)
        b_lists.append(b)
        elasticity_lists.append(elasticity)
        r_squared_lists.append(r_squared)

    # Return the average relative error
    mean_relative_error = np.mean(relative_errors)
    mean_a = np.mean(a_lists)
    mean_b = np.mean(b_lists)
    mean_elasticity = np.mean(elasticity_lists)
    mean_r_squared = np.mean(r_squared_lists)

    return mean_relative_error, mean_a, mean_b, mean_elasticity, mean_r_squared

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
    else:
        # Add columns for the best model
        results['best_model'] = best_model
        results['best_model_a'] = results[best_model + '_a']
        results['best_model_b'] = results[best_model + '_b']
        results['best_model_r2'] = results[best_model + '_r2']
        results['best_mean_relative_error'] = results[best_model + '_mean_relative_error']
        results['best_model_elasticity'] = results[best_model + '_elasticity']

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
