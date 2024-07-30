"""This module contains data classes for the S3 module, used to reduce the number of parameters."""

from pydantic import BaseModel


class UploadToS3(BaseModel):
    """Data class for upload_to_s3 keyword arguments.

    Attributes:
        s3_dir (str): The S3 directory where the file should be stored.
        file_name (str): The name to assign to the file in S3.
        is_serializable (bool, optional): If True, the object will be serialized before upload.
            Default is False.
    """

    s3_dir: str
    file_name: str
    is_serializable: bool = False
