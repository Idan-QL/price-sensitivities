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
    "storage_location": {
        "flag": "-d",
        "default": "",
        "type": str,
        "help": "Bucket name or Data-Center ('us' or 'eu').",
        "required": True,
    },
    "is_qa_run": {
        "flag": "-q",
        "default": "True",
        "type": literal_eval,
        "help": "Read/Write results to QA/staging or production environment. "
        "May be `True` (QA/staging run) or `False` (production run)",
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
