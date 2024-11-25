SELECT uid,
      strings_map['best_model'] as best_model
FROM AwsDataCatalog.ds_sandbox.models_monitoring
WHERE
    project_name='elasticity'
    and client_key = '{client_key}'
    and channel = '{channel}'
    and strings_map['data_date'] = '{previous_data_date}'
    and strings_map['result_to_push'] = 'True'