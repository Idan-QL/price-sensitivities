"""Module of configurator."""

from typing import Optional

from pydantic import BaseModel


class DataColumns(BaseModel):
    """Represents the data columns used in the application.

    Attributes:
        uid (Optional[str]): The column name for the UID.
        price (Optional[str]): The column name for the price.
        quantity (Optional[str]): The column name for the quantity.
        date (Optional[str]): The column name for the date.
        weight (Optional[str]): The column name for the weight.
    """

    uid: Optional[str] = "uid"
    price: Optional[str] = "round_price"
    quantity: Optional[str] = "units"
    date: Optional[str] = "date"
    weight: Optional[str] = "days"


class PreprocessingParameters(BaseModel):
    """Represents the parameters used for preprocessing data.

    Attributes:
        price_changes (int): The number of price changes to consider.
        threshold (float): The threshold value for determining significant changes.
        min_conversions_days (int): The minimum number of days for considering conversions.
    """

    price_changes: int = 5
    threshold: float = 0.01
    min_conversions_days: int = 10
