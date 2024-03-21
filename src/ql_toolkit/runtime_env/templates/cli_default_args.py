"""Example for CLI arguments default dict definition."""

from ast import literal_eval

args_dict = {
    "aws_key": {
        "flag": "-k",
        "default": "",
        "type": str,
        "help": "AWS access key",
        "required": False,
    },
    "aws_secret": {
        "flag": "-s",
        "default": "",
        "type": str,
        "help": "AWS secret key",
        "required": False,
    },
    "project_name": {
        "flag": "-p",
        "default": "",
        "type": str,
        "help": "Project name",
        "required": True,
    },
    "local": {
        "flag": "-l",
        "default": "False",
        "type": literal_eval,
        "help": "Write results to local 'artifacts' " "folder or to S3",
        "required": False,
    },
    "data_center": {
        "flag": "-d",
        "default": "us",
        "type": str,
        "help": "Data-Center ('us' or 'eu')",
        "required": False,
    },
    "config": {
        "flag": "-c",
        "default": "config",
        "type": str,
        "help": "Config file location in s3",
        "required": False,
    },
}
