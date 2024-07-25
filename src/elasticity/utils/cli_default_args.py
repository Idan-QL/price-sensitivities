"""The default values of the command line argument for this project."""

from ast import literal_eval

args_kv = {
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
        "default": "elasticity",
        "type": str,
        "help": "Project name",
        "required": True,
    },
    "data_center": {
        "flag": "-d",
        "default": "us",
        "type": str,
        "help": "Data-Center ('us' or 'eu')",
        "required": False,
    },
    "local": {
        "flag": "-l",
        "default": "False",
        "type": literal_eval,
        "help": "Write results to local 'artifacts' " "folder or to S3",
        "required": False,
    },
    "is_qa_run": {
        "flag": "-qa",
        "default": "False",
        "type": literal_eval,
        "help": "Read config from QA spreadsheet",
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
