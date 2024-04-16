# Elasticity

Elasticity : Measurement of change in quantity demanded due to the change in the price. An elasticity of price of -3 indicates that for every 1% increase in price, the quantity demanded decreases by 3%.

This project is designed to calculate elasticity accross clients and channels, write elasticity attributs and reports

## Main.py

The main script in this project is `main.py`. This script performs several key tasks by client_key and clannel:

1. **Process Data**: The `preprocessing.read_and_preprocess` function is used to read and process the data. It is reading one month to one year of data  

2. **Elasticity calculation**: The `run_experiment_for_uids_parallel` function is calculate elasticity and output a dataframe with the resilts

3. **Saving results**: The `write_dataframe_to_s3` function is used to write results to data_science/eval_results/elasticity/ as elasticity_{client_key}_{channel}_{end_date}.csv 

4. **Top uid elasticity graph saving**: The `run_save_graph_top_10` function is used to save the 10 best elasticity results graph to data_science/eval_results/elasticity/graphs/{client_key}/{channel}/ as {uid}_{end_date}.png 

5. **Writing Action List**: The `write_actions_list` function is used to write the attributs. Attributs are:
- qlia_elasticity_param1
- qlia_elasticity_param2
- qlia_elasticity_demand_model
- qlia_product_elasticity
- qlia_elasticity_calc_date

6. **Reporting**: The `report.add_run` function is used to add a run to the report. The final report is saved to 
data_science/eval_results/elasticity/ as elasticity_report_{end_date}.csv

7. **Distribution graph Saving**: The `write_graphs.save_distribution_graph` function is used to save the elasticity distribution graph

8. **Error Handling**: If an error occurs during the processing of a client key and channel, the error is logged and an error run is added to the report using the `report.add_error_run` function.

## Dependencies

- Python 3.x
- Poetry
- Dependencies specified in pyproject.toml, poetry.lock and requirements.txt

## File Structure

- src/: Contains the main Python scripts for data processing (main.py) and related modules.
- report/: Module for generating reports and logging errors.
- elasticity/: Contains modules for elasticity analysis and model execution.
- ql_toolkit/: Toolkit for various utility functions and interactions with the environment.
- requirements.txt: Specifies project dependencies.

## Usage

To run the main script, navigate to the project directory and run the following command:

```bash
python src/main.py -d us -c config -p elasticity 
```
config in s3: data_science/sagemaker/config_files/elasticity/
- qa: config_qa
- inventory: config-inventory
