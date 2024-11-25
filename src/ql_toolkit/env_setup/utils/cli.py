"""This module contains the CliArgsParser class, which is used to parse the CLI arguments."""

import argparse
import json
import logging
from os import path
from sys import argv
from sys import exit as sys_exit
from typing import Optional

"""
Usage: in (typically) the python entry-point (e.g. in main.py), add the following lines
(change kv_demo as you wish):

from ast import literal_eval  # If a boolean argument is needed. Then the arg must be given as
'True' or 'False' (e.g. $: python main.py --local True)

kv_demo = {
    "aws_key": {"flag": "-k", "default": "", "type": str, "help": "AWS access key",
    "required": False},
    "aws_secret": {"flag": "-s", "default": "", "type": str, "help": "AWS secret key",
    "required": False},
    "local": {"flag": "-l", "default": "False", "type": literal_eval, "help": "boolean type",
    "required": False},
    "num": {"flag": "-n", "default": 2, "type": int, "help": "int type", "required": False},
    "pct": {"flag": "-p", "default": 0.79, "type": float, "help": "float type", "required": False},
    "vals": {"flag": "-v", "default": ['a', 'b'], "type": list, "help": "list type",
    "required": False},
    "kv": {"flag": "-d", "default": {'x': '1', 'y': '2'}, "type": dict, "help": "dict type",
    "required": False}
}

def run():
    parser = cli.Parser(kv=args_kv)
    parser.set_args_dict()
    args_dict = parser.get_args_dict()
"""


class CliArgsParser:
    """Factory class for argparse.ArgumentParser."""

    def __init__(self, kv: dict[str, dict], parser_name: str = "") -> None:
        """Constructor for the CliArgsParser class.

        Args:
            kv (dict[str, dict]): The key-value pairs of the arguments
            parser_name (str): The name of the parser. Defaults to "".
        """
        self.kv = kv
        self.parser_name = parser_name
        self.parser = None

    def add_args(self) -> None:
        """This function adds the arguments to the parser.

        Returns:
            None
        """
        for arg_name, arg_vals in self.kv.items():
            args = [f"--{arg_name}", arg_vals["flag"]]
            kwargs = {key: arg_vals[key] for key in ["default", "help", "type", "required"]}
            self.parser.add_argument(*args, **kwargs)

    def set_parser(self) -> None:
        """This function sets the parser.

        Returns:
            None
        """
        self.parser = argparse.ArgumentParser(self.parser_name)
        self.add_args()

    def set_args_dict_from_cli(self) -> dict:
        """This function sets the arguments from the command line.

        Returns:
            dict: The arguments dictionary
        """
        self.set_parser()
        parsed_args = self.parser.parse_args()
        return vars(parsed_args)


class Parser(CliArgsParser):
    """This class is used to parse the CLI arguments."""

    def __init__(
        self,
        kv: dict[str, dict],
        parser_name: str = "",
        cli_args_conf_path: str = "run_files",
    ) -> None:
        """Constructor for the Parser class.

        Args:
            kv (dict[str, dict]): The key-value pairs of the arguments
            parser_name (str): The name of the parser. Defaults to "".
            cli_args_conf_path (str, default "run_files"): The path to the CLI arguments
                configuration file.
        """
        super().__init__(kv, parser_name)
        # Notice that for Sagemaker, the cli args are given as a json file
        self.cli_args_conf_path = cli_args_conf_path
        self.args_dict = None

    def get_args_dict(self) -> Optional[dict]:
        """This function gets the arguments dictionary."""
        return self.args_dict

    def set_args_dict(self, args_dict: Optional[dict] = None) -> None:
        """This function sets the arguments dictionary.

        Args:
            args_dict (Optional[dict]): The arguments' dictionary. Defaults to None.

        Returns:
            None
        """
        if args_dict:
            self.args_dict = args_dict
            self.validate_args()
            return

        self.args_dict = self.read_args_from_cli_or_json()

        if self.args_dict is None:
            sys_exit("No arguments found. Exiting!")
        self.validate_args()

    def read_args_from_cli_or_json(self) -> Optional[dict]:
        """Reads arguments from the CLI or configuration file based on the environment.

        Returns:
            dict: The arguments dictionary loaded from CLI or config, or None if not found.
        """
        if len(argv) > 1:
            print("Reading arguments from argv")
            return self.set_args_dict_from_cli()

        return self.read_args_from_json()

    def read_args_from_json(self) -> Optional[dict]:
        """Reads arguments from configuration files depending on the execution environment.

        Returns:
            dict: The arguments dictionary if file is found and read, otherwise None.
        """
        config_paths = [
            "/opt/ml/input/config",  # Path in container
            self.cli_args_conf_path,  # Local path
        ]
        for config_dir in config_paths:
            config_file_path = path.join(config_dir, "hyperparameters.json")
            if path.exists(config_file_path):
                try:
                    with open(config_file_path, "r", encoding="utf-8") as file:
                        print(f"Reading arguments from {config_file_path}")
                        return json.load(file)
                except FileNotFoundError as err:
                    logging.error(f"Error caught: {err}")
                    sys_exit("Exiting: The hyperparameters.json is not found...\n")
        return None

    def set_bool_arg(self, key: str) -> None:
        """This function sets up a bool argument.

        Args:
            key (str): The key of the argument

        Returns:
            None
        """
        if not isinstance(self.args_dict[key], bool):
            self.args_dict[key] = self.args_dict[key] in {"True", "true"}

    def set_int_arg(self, key: str) -> None:
        """This function sets up an int argument.

        Args:
            key (str): The key of the argument

        Returns:
            None
        """
        if not isinstance(self.args_dict[key], int):
            try:
                self.args_dict[key] = int(self.args_dict[key])
            except ValueError as err:
                sys_exit(f"ValueError caught when trying to set up an int argument: {err}")

    def set_float_arg(self, key: str) -> None:
        """This function sets up a float argument.

        Args:
            key (str): The key of the argument

        Returns:
            None
        """
        if not isinstance(self.args_dict[key], float):
            try:
                self.args_dict[key] = float(self.args_dict[key])
            except ValueError as err:
                sys_exit(f"ValueError caught when trying to set up an float argument: {err}")

    def validate_args(self) -> None:
        """This function validates the arguments given in the hyperparameters.json file.

        It sets default values for arguments that are not already defined and adjusts
        their values based on their type by calling the appropriate setter function.

        Returns:
            None
        """
        # Mapping of types to their respective setter functions
        type_function_map = {
            bool: self.set_bool_arg,
            int: self.set_int_arg,
            float: self.set_float_arg,
        }

        for arg_name, arg_vals in self.kv.items():
            # Set default value if arg_name is not in args_dict
            if arg_name not in self.args_dict:
                self.args_dict[arg_name] = arg_vals["default"]

            # Retrieve the function based on the type and call it
            arg_type = arg_vals["type"]
            setter_function = type_function_map.get(arg_type)
            if setter_function:
                setter_function(arg_name)
