"""This module contains the test for setup.py in qltoolkit."""

from sys import path as sys_path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys_path.append("../src")
from ql_toolkit.config.runtime_config import app_state
from ql_toolkit.runtime_env.setup import get_config_from_sheet


@patch("ql_toolkit.runtime_env.setup.get_google_sheet_data")
def test_get_config_from_sheet(mock_get_google_sheet_data: MagicMock) -> None:
    """Test case for the get_config_from_sheet function.

    This test case verifies that the get_config_from_sheet function returns
    the expected configuration.

    Args:
        mock_get_google_sheet_data: A mock object for the get_google_sheet_data function.

    Returns:
        None
    """
    spreadsheet_name = "DS_PROD_CONFIG"
    sheet_name = "elasticity"
    data_center = "eu"
    app_state.project_name = "elasticity"
    app_state.bucket_name = "quicklizard"

    # Mock the return value of get_google_sheet_data
    mock_get_google_sheet_data.return_value = pd.DataFrame(
        {
            "data_center": ["eu", "eu", "eu"],
            "client_key": ["mock_cl1", "mock_cl2", "mock_cl3"],
            "channels": ["['default']", "['default']", "['default']"],
            "attributes": ["attr1", "attr2", "attr3"],
        }
    )

    # Call the function under test
    config = get_config_from_sheet(spreadsheet_name, sheet_name, data_center)

    # Assert the expected result
    expected_config = {
        "client_keys": {
            "mock_cl1": {"channels": ["default"], "attr_name": ["attr1"]},
            "mock_cl2": {"channels": ["default"], "attr_name": ["attr2"]},
            "mock_cl3": {"channels": ["default"], "attr_name": ["attr3"]},
        }
    }
    assert config == expected_config


@patch("ql_toolkit.runtime_env.setup.get_google_sheet_data")
def test_get_config_from_sheet_duplicate(mock_get_google_sheet_data: MagicMock) -> None:
    """Test case for checking if duplicate entries are handled correctly.

    Raises:
        SystemExit: If duplicate entries are found for client_keys.

    Returns:
        None
    """
    spreadsheet_name = "DS_PROD_CONFIG"
    sheet_name = "elasticity"
    data_center = "eu"
    app_state.project_name = "elasticity"
    app_state.bucket_name = "quicklizard"

    # Mock the return value of get_google_sheet_data
    mock_get_google_sheet_data.return_value = pd.DataFrame(
        {
            "data_center": ["eu", "eu", "eu"],
            "client_key": ["mock_cl1", "mock_cl1", "mock_cl3"],
            "channels": ["['default']", "['default']", "['default']"],
            "attributes": ["attr1", "attr2", "attr3"],
        }
    )

    with pytest.raises(SystemExit) as excinfo:
        get_config_from_sheet(spreadsheet_name, sheet_name, data_center)

    assert (
        excinfo.value.code
        == "Spreadsheet config: Duplicate entries found for client_keys: ['mock_cl1']"
    )
