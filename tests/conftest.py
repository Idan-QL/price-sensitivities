"""conftest ofr pytest fixture."""

import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture providing sample data for tests.

    those data are wollowing a linear demand curve with elasticity of -1 at median price of 30
    """
    return pd.DataFrame(
        {
            "price": [10, 20, 30, 40, 50],
            "quantity": [100, 80, 60, 40, 20],
            "days": [1, 2, 3, 4, 5],
            "last_price": [30, 30, 30, 30, 30],
            "last_date": ["20240301", "20240301", "20240301", "20240301", "20240301"],
        }
    )
