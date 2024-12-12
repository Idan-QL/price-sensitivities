"""Module of configurator."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DateRange(BaseModel):
    """Represents the date range for filtering data."""

    start_date: str = Field(..., description="The start date in format YYYY-MM-DD")
    end_date: str = Field(..., description="The end date in format YYYY-MM-DD")


class DataFetchParameters(BaseModel):
    """Parameters related to fetching data."""

    client_key: str = Field(..., description="The client key for identifying data retrieval scope.")
    channel: str = Field(..., description="The specific channel used for data filtering.")
    attr_names: Optional[List[str]] = Field(
        None, description="Optional attribute name for additional filtering."
    )
    source: Literal["analytics", "product_extended_daily", "product_transaction"] = Field(
        "analytics",
        description="The data source:analytics, product_extended_daily, or transaction.",
    )
    uids_to_filter: Optional[List[str]] = Field(
        None, description="Optional list of UIDs to filter the data by."
    )


class PreprocessingParameters(BaseModel):
    """Represents the parameters used for preprocessing data."""

    price_changes: int = Field(5, description="Number of price changes required")
    threshold: float = Field(0.01, description="Threshold value for detecting price changes")
    min_conversions_days: int = Field(10, description="Minimum number of days for conversions")


class DataColumns(BaseModel):
    """Represents the data columns used in the application."""

    uid: Optional[str] = Field("uid", description="The column name for the UID")
    round_price: Optional[str] = Field(
        "round_price", description="The column name for the round price"
    )
    shelf_price: Optional[str] = Field(
        "shelf_price", description="The column name for the shelf price"
    )
    quantity: Optional[str] = Field("units", description="The column name for the quantity")
    revenue: Optional[str] = Field("revenue", description="The column name for the revenue")
    date: Optional[str] = Field("date", description="The column name for the date")
    weight: Optional[str] = Field("days", description="The column name for the weight")
    inventory: Optional[str] = Field("inventory", description="The column name for the inventory")
    outlier_quantity: Optional[str] = Field(
        "outlier_quantity", description="The column name for the outlier_quantity"
    )
