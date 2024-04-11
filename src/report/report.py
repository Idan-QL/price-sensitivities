from typing import List, Dict
import pandas as pd

def add_run(data_report: List,
                       client_key: str,
                       channel: str,
                       total_uid: int,
                       df_results: pd.DataFrame,
                       runtime: float,
                       max_elasticity=3.8,
                       min_elasticity=-3.8) -> None:
    """
    Append data to the data report list.

    Parameters:
        data_report (List[Dict[str, Any]]): The list to append data to.
        client_key (Any): The client key.
        channel (Any): The channel.
        total_end_date_uid (Any): The total end date UID.
        df_results_quality (DataFrame): The DataFrame containing quality results.
        df_results (DataFrame): The DataFrame containing results.

    Returns:
        None
    """

    df_results_quality = df_results[df_results["quality_test"] == True]
    best_model_counts = df_results_quality['best_model'].value_counts()

    data_report.append({
        "client_key": client_key,
        "channel": channel,
        "total_uid": total_uid,
        "uid_with_elasticity": len(df_results_quality),
        "percentage_from_total": round(len(df_results)/total_uid * 100, 1),
        "percentage_from_total_with_data": round(len(df_results)/len(df_results) * 100, 1),
        "uid_with_data_for_elasticity": len(df_results),
        "uid_with_elasticity_less_than_minus3.8": len(df_results_quality[df_results_quality.best_model_elasticity < min_elasticity]),
        "uid_with_elasticity_moreorequal_minus3.8_less_than_minus1": len(df_results_quality[(df_results_quality.best_model_elasticity >= min_elasticity) & (df_results_quality.best_model_elasticity < -1)]),
        "uid_with_elasticity_moreorequal_minus1_less_than_0": len(df_results_quality[(df_results_quality.best_model_elasticity >= -1) & (df_results_quality.best_model_elasticity < 0)]),
        "uid_with_elasticity_moreorequal_0_less_than_1": len(df_results_quality[(df_results_quality.best_model_elasticity >= 0) & (df_results_quality.best_model_elasticity < 1)]),
        "uid_with_elasticity_moreorequal_1_less_than_3.8": len(df_results_quality[(df_results_quality.best_model_elasticity >= 1) & (df_results_quality.best_model_elasticity < max_elasticity)]),
        "uid_with_elasticity_more_than_3.8": len(df_results_quality[df_results_quality.best_model_elasticity > max_elasticity]),
        "best_model_power_count": best_model_counts.get('power', 0),
        "best_model_exponential_count": best_model_counts.get('exponential', 0),
        "best_model_linear_count": best_model_counts.get('linear', 0),
        "runtime_duration": runtime,
        "error": False
    })

    return data_report


def add_error_run(data_report: List,
                  client_key: str,
                  channel: str) -> None:
    """
    Append data to the data report list.

    Parameters:
        data_report (List[Dict[str, Any]]): The list to append data to.
        client_key (Any): The client key.
        channel (Any): The channel.
        total_end_date_uid (Any): The total end date UID.
        df_results_quality (DataFrame): The DataFrame containing quality results.
        df_results (DataFrame): The DataFrame containing results.

    Returns:
        None
    """

    data_report.append({
        "client_key": client_key,
        "channel": channel,
        "total_uid": None,
        "uid_with_elasticity":None ,
        "uid_with_data_for_elasticity": None,
        "uid_with_elasticity_less_than_minus3.8": None,
        "uid_with_elasticity_moreorequal_minus3.8_less_than_minus1": None,
        "uid_with_elasticity_moreorequal_minus1_less_than_0": None,
        "uid_with_elasticity_moreorequal_0_less_than_1": None,
        "uid_with_elasticity_moreorequal_1_less_than_3.8": None,
        "uid_with_elasticity_more_than_3.8": None,
        "best_model_power_count": None,
        "best_model_exponential_count": None,
        "best_model_linear_count": None,
        "runtime_duration": None,
        "error": True
    })

    return data_report
