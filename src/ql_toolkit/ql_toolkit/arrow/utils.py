"""Utility functions for working with Arrow data structures."""

import pandas as pd
import pyarrow as pa


def cast_df_as_arrow_table(data_df: pd.DataFrame, schema: pa.schema) -> pa.Table:
    """Casts a pandas DataFrame as an Arrow Table.

    Args:
        data_df (pd.DataFrame): The pandas DataFrame to cast.
        schema (Dict): The schema of the Arrow Table.

    Returns:
        pa.Table: The Arrow Table representing the data in the pandas DataFrame.
    """
    return pa.Table.from_pandas(df=data_df, schema=schema)
