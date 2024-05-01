"""Module of elasticity_plotter."""

from typing import Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# from elasticity.data.preprocessing import preprocess_by_day


class ElasticityPlotter:
    """A class to plot various types of graphs.

    Attributes:
    - uid_col (str): Column name for the UID.
    - price_col (str): Column name for the price data.
    - quantity_col (str): Column name for the quantity data.
    - median_price_col (str): Column name for the median price.
    - median_quantity_col (str): Column name for the median quantity.
    - date_col (str): Column name for the date.
    """

    def __init__(
        self,
        uid_col: str = "uid",
        price_col: str = "price",
        quantity_col: str = "units",
        median_price_col: str = "uid_price_median",
        median_quantity_col: str = "median_sale_per_day",
        date_col: str = "date",
    ) -> None:
        """Initializes the ElasticityPlotter with default parameters."""
        self.uid_col = uid_col
        self.price_col = price_col
        self.quantity_col = quantity_col
        self.median_price_col = median_price_col
        self.median_quantity_col = median_quantity_col
        self.date_col = date_col

    def jointplot_with_median(
        self,
        dfplot: pd.DataFrame,
        median_price: float,
        median_quantity: float,
        title: str = "",
        color: str = "yellow",
    ) -> None:
        """Generate a joint plot with median lines.

        Parameters:
        - dfplot (DataFrame): DataFrame containing the data.
        - median_price (float): Median value for the price.
        - median_quantity (float): Median value for the quantity.
        - title (str): Title for the plot (default is '').
        - color (str): Color for the plot (default is "yellow").

        Returns:
        None
        """
        h = sns.jointplot(
            data=dfplot, x=self.price_col, y=self.quantity_col, color=color
        )
        h.refline(x=median_price, y=median_quantity, color="red")
        h.figure.suptitle(title)
        h.figure.tight_layout()
        h.figure.subplots_adjust(top=0.95)  # Reduce plot to make room
        plt.show()

    def plot_curves(self, dfplot: pd.DataFrame, uid: str) -> None:
        """Plot various curves for a specific UID.

        Parameters:
        - df (DataFrame): DataFrame containing the data.
        - uid (str or int): Unique identifier for the data.

        Returns:
        None
        """
        df_uid = dfplot[dfplot[self.uid_col] == uid]
        # df_by_day_uid, df_by_price_norm_uid = preprocess_by_day(df_uid)

        if self.median_price_col not in df_uid.columns:
            df_uid[self.median_price_col] = df_uid[self.price_col].median()
        if self.median_quantity_col not in df_uid.columns:
            df_uid[self.median_quantity_col] = df_uid[self.quantity_col].median()

        median_price_uid = df_uid[self.median_price_col].iloc[0]
        median_quantity_uid = df_uid[self.median_quantity_col].iloc[0]

        self.jointplot_with_median(
            df_uid,
            median_price_uid,
            median_quantity_uid,
            title="Data by price normalized - UID: " + str(uid),
            color="orange",
        )

    def plot_curves_by_price_norm(self, dfplot: pd.DataFrame, uid: str) -> None:
        """Plot curves by price normalization for a specific UID.

        Parameters:
        - dfplot (DataFrame): DataFrame containing the data.
        - uid (str or int): Unique identifier for the data.

        Returns:
        None
        """
        if self.median_price_col not in dfplot.columns:
            dfplot[self.median_price_col] = dfplot[self.price_col].median()
        if self.median_quantity_col not in dfplot.columns:
            dfplot[self.median_quantity_col] = dfplot[self.quantity_col].median()

        df_uid = dfplot[dfplot[self.uid_col] == uid]
        # df_by_day_uid, df_by_price_norm_uid = preprocess_by_day(df_uid)

        median_price_uid = df_uid[self.median_price_col].iloc[0]
        median_quantity_uid = df_uid[self.median_quantity_col].iloc[0]

        self.jointplot_with_median(
            df_uid,
            median_price_uid,
            median_quantity_uid,
            title="Data by price normalized - UID: " + str(uid),
            color="orange",
        )

    def plot_price_and_units_for_uid_one_graph(
        self, dfplot: pd.DataFrame, uid: str
    ) -> None:
        """Plot price and units for a specific UID in one graph.

        Parameters:plot_price_and_units_for_uid_one_graph
        - dfplot (DataFrame): DataFrame containing the data.
        - uid (str or int): Unique identifier for the data.

        Returns:
        None
        """
        # df_by_day_uid, _ = preprocess_by_day(df[df[self.uid_col] == uid])

        dfplot = dfplot[dfplot[self.uid_col] == uid]

        dfplot["date"] = pd.to_datetime(dfplot[self.date_col])

        dfplot = dfplot.sort_values(by=self.date_col)

        if self.median_price_col not in dfplot.columns:
            dfplot[self.median_price_col] = dfplot[self.price_col].median()
        if self.median_quantity_col not in dfplot.columns:
            dfplot[self.median_quantity_col] = dfplot[self.quantity_col].median()

        median_price_uid = dfplot[self.median_price_col].iloc[0]
        median_quantity_uid = dfplot[self.median_quantity_col].iloc[0]

        fig, ax1 = plt.subplots()

        sns.lineplot(data=dfplot, x="date", y=self.price_col, ax=ax1)
        ax1.set_ylabel("Price from Revenue", color="tab:blue")

        ax1.axhline(
            y=median_price_uid, color="tab:blue", linestyle="--", label="Median price"
        )

        ax2 = ax1.twinx()

        sns.lineplot(
            data=dfplot,
            x="date",
            y=self.quantity_col,
            ax=ax2,
            color="tab:orange",
        )
        ax2.set_ylabel("Units", color="tab:orange")

        print(dfplot.date.min(), dfplot.date.max())

        ax2.axhline(
            y=median_quantity_uid,
            color="tab:orange",
            linestyle="--",
            label="Median Units",
        )

        ax1.legend(loc="upper left")
        ax2.legend(loc="lower left")

        plt.title(f"Price and Units for UID {uid}")

        ax1.set_xlabel("Date")
        ax1.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax1.xaxis.get_major_locator())
        )
        plt.show()

    def plot_price_and_units_for_uid_2_graphs(
        self, dfplot: pd.DataFrame, uid: Union[str, int]
    ) -> None:
        """Plot price and units for a specific UID in two separate graphs.

        Parameters:
        - dfplot (DataFrame): DataFrame containing the data.
        - uid (str or int): Unique identifier for the data.

        Returns:
        None
        """
        # df_by_day_uid, _ = preprocess_by_day(df[df[self.uid_col] == uid])

        df_uid = dfplot[dfplot[self.uid_col] == uid]

        df_uid[self.date_col] = pd.to_datetime(df_uid[self.date_col])

        df_uid = df_uid.sort_values(by=self.date_col)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        if self.median_price_col not in df_uid.columns:
            df_uid[self.median_price_col] = df_uid[self.price_col].median()
        if self.median_quantity_col not in df_uid.columns:
            df_uid[self.median_quantity_col] = df_uid[self.quantity_col].median()

        median_price_uid = df_uid[self.median_price_col].iloc[0]
        median_quantity_uid = df_uid[self.median_quantity_col].iloc[0]

        sns.lineplot(data=df_uid, x="price", y=self.price_col, ax=ax1)
        ax1.set_ylabel("Price from Revenue", color="tab:blue")

        ax1.axhline(
            y=median_price_uid, color="tab:blue", linestyle="--", label="Median price"
        )
        ax1.legend(loc="upper left")

        sns.lineplot(
            data=df_uid,
            x="price",
            y=self.quantity_col,
            ax=ax2,
            color="tab:orange",
        )
        ax2.set_ylabel("Units", color="tab:orange")

        ax2.axhline(
            y=median_quantity_uid,
            color="tab:orange",
            linestyle="--",
            label="Median Units",
        )
        ax2.legend(loc="upper left")

        plt.title(f"Price and Units for UID {uid}")
        ax2.set_xlabel("Date")
        ax2.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator())
        )

        plt.tight_layout()
        plt.show()
