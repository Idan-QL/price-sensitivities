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
        """This function initializes the class."""
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
        """This function initializes the class."""
        super().__init__(kv, parser_name)
        # Notice that for Sagemaker, the cli args are given as a json file
        self.cli_args_conf_path = cli_args_conf_path
        self.args_dict = None

    def get_args_dict(self) -> Optional[dict]:
        """This function gets the arguments dictionary."""
        return self.args_dict

    def set_args_dict(self, args_dict: dict = None) -> None:
        """This function sets the arguments dictionary.

        Args:
            args_dict (dict): The arguments' dictionary. Defaults to None.

        Returns:
            None
        """
        if args_dict:
            self.args_dict = args_dict
        else:
            if len(argv) > 1:
                print("Reading arguments from argv")
                self.args_dict = self.set_args_dict_from_cli()
            else:
                # Set reading arguments from an Estimator container
                if path.exists("/opt/ml"):
                    print("Reading arguments from /opt/ml/input/config (inside container)")
                    # The config folder contains the "cli args" in json format
                    self.cli_args_conf_path = "/opt/ml/input/config"
                # Allow reading arguments from an Estimator container in 'local' mode
                elif path.exists(self.cli_args_conf_path):
                    print(f"Reading arguments from {self.cli_args_conf_path} (on local machine)")
                try:
                    with open(path.join(self.cli_args_conf_path, "hyperparameters.json")) as f:
                        self.args_dict = json.load(f)
                except FileNotFoundError as err:
                    logging.error("Error caught: %s", err)
                    sys_exit(
                        "In runtime_env/cli.py: The hyperparameters.json is not "
                        "found and no CLI args are given...\n"
                        "Exiting!"
                    )

        self.validate_args()
        if self.args_dict is None:
            sys_exit("No arguments found. Exiting!")

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

        Returns:
            None
        """
        for arg_name, arg_vals in self.kv.items():
            if arg_name not in self.args_dict:
                self.args_dict[arg_name] = arg_vals["default"]
            if arg_vals["type"] == bool:
                self.set_bool_arg(arg_name)
            elif arg_vals["type"] == int:
                self.set_int_arg(arg_name)
            elif arg_vals["type"] == float:
                self.set_float_arg(arg_name)
