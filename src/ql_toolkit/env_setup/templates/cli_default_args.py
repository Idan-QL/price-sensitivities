"""The REQUIRED command line argument for the project."""

from ast import literal_eval

default_args_dict = {
    "aws_key": {
        "flag": "-k",
        "default": "",
        "type": str,
        "help": "AWS access key. Default is an empty `str`.",
        "required": False,
    },
    "aws_secret": {
        "flag": "-s",
        "default": "",
        "type": str,
        "help": "AWS secret key. Default is an empty `str`.",
        "required": False,
    },
    "project_name": {
        "flag": "-p",
        "default": "",
        "type": str,
        "help": "Project name.",
        "required": True,
    },
    "storage_location": {
        "flag": "-d",
        "default": "",
        "type": str,
        "help": "Bucket name or generic storage location ('us' or 'eu') for reading data.",
        "required": True,
    },
    "config": {
        "flag": "-c",
        "default": "",
        "type": str,
        "help": "Config file name (without file post-fix. E.g. `qa_config`).",
        "required": True,
    },
    "is_qa_run": {
        "flag": "-q",
        "default": "",
        "type": literal_eval,
        "help": "Write results to QA/staging or production environment. "
        "May be `True` (QA/staging run) or `False` (production run)",
        "required": True,
    },
    # Add other arguments as needed
}
