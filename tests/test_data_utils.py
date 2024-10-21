"""Test module of model/run_model."""

import pandas as pd

from elasticity.data.configurator import DataColumns
from elasticity.data.utils import summarize_price_history


def test_summarize_price_history() -> None:
    """Test summarize_price_history."""
    data = {
        "uid": ["A", "A", "A", "B", "B", "C"],
        "date": [
            "2023-01-01",
            "2023-02-01",
            "2023-03-01",
            "2023-01-01",
            "2023-02-01",
            "2023-01-01",
        ],
        "shelf_price": [10.0, 12.0, 11.0, 20.0, 22.0, 30.0],
    }
    input_df = pd.DataFrame(data)
    input_df["date"] = pd.to_datetime(input_df["date"])

    # Define data columns
    data_columns = DataColumns(uid="uid", date="date", shelf_price="shelf_price")

    # Expected output
    expected_data = {
        "uid": ["A", "B", "C"],
        "last_date": ["2023-03-01", "2023-02-01", "2023-01-01"],
        "first_date": ["2023-01-01", "2023-01-01", "2023-01-01"],
        "min_price": [10.0, 20.0, 30.0],
        "max_price": [12.0, 22.0, 30.0],
        "last_price": [11.0, 22.0, 30.0],
    }

    expected_df = pd.DataFrame(expected_data)
    expected_df["last_date"], expected_df["first_date"] = pd.to_datetime(
        expected_df["last_date"]
    ), pd.to_datetime(expected_df["first_date"])

    result_df = summarize_price_history(input_df, data_columns)

    pd.testing.assert_frame_equal(result_df, expected_df)
