"""This module contains utility functions for plotting data."""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

common_kwargs = [
    "df",
    "x_col",
    "y_col",
    "title",
    "hue",
    "style",
    "title_font_size",
    "plot_font_scale",
    "fig_size",
    "save_fig",
    "file_name",
]


def validate_inputs(
    ptype: str, df: pd.DataFrame | pl.DataFrame | pl.LazyFrame, x_col: str
) -> None:
    """Validate the inputs for the plot.

    Args:
        ptype (str): The type of plot. It must be a string.
        df (pd.DataFrame | pl.DataFrame | pl.LazyFrame): The DataFrame that contains the data
            to plot.
        x_col (str): The name of the column that holds the horizontal axis data.
            It must be a string.

    Returns:
        None
    """
    if not isinstance(ptype, str):
        raise ValueError("The value of 'ptype' must be a string")
    if not isinstance(df, (pd.DataFrame, pl.DataFrame, pl.LazyFrame)):
        raise ValueError("The DataFrame must be a pandas or polars DataFrame")
    if not isinstance(x_col, str):
        raise ValueError("The value of 'x_col' must be a string")


def convert_df(plot_df: pd.DataFrame | pl.DataFrame | pl.LazyFrame) -> pd.DataFrame:
    """Convert polars DataFrame to pandas DataFrame if necessary.

    Args:
        plot_df (pd.DataFrame | pl.DataFrame | pl.LazyFrame): The DataFrame to convert.

    Returns:
        pd.DataFrame: The converted DataFrame.
    """
    if isinstance(plot_df, pl.LazyFrame):
        plot_df = plot_df.collect()
    if isinstance(plot_df, pl.DataFrame):
        plot_df = plot_df.to_pandas()
    return plot_df


def set_defaults(
    kwargs: dict[str, any], df: pd.DataFrame, x_col: str
) -> dict[str, any]:
    """Set default values for kwargs.

    Args:
        kwargs (dict): The keyword arguments for the plot.
        df (pd.DataFrame): The DataFrame that contains the data to plot.
        x_col (str): The name of the column that holds the horizontal axis data.

    Returns:
        dict: The updated dictionary of arguments for the plot.
    """
    sns.set(font_scale=kwargs.get("plot_font_scale", 1.0))
    plt.figure(figsize=kwargs.get("fig_size", (12, 8)))

    defaults = {
        "y_col": None,
        "hue": None,
        "style": None,
        "title": f"{kwargs.get('ptype', '').title()} plot",
        "title_font_size": kwargs.get("title_font_size", 14.0),
        "x_label": x_col,
        "y_label": kwargs.get("y_col"),
        "log_scale": (False, False),
        "x_range": (None, None),
        "y_range": (None, None),
        "save_fig": False,
        "df": df,
        "x_col": x_col,
    }

    # Update defaults with any kwargs provided, prioritizing kwargs values
    return {**defaults, **kwargs}


def universal_plot_args(
    ptype: str,
    pd_df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    x_col: str,
    **kwargs: dict,
) -> dict:
    """Prepare args for various types of plots, simplifying input validation and setting defaults.

    Args:
        ptype (str): The type of plot. It must be a string.
        pd_df (pd.DataFrame | pl.DataFrame | pl.LazyFrame): The DataFrame that contains the data
            to plot.
        x_col (str): The name of the column that holds the horizontal axis data.
        **kwargs: Additional keyword arguments for the plot.
            These can include 'y_col', 'hue', 'style', 'title', 'title_font_size',
            'plot_font_scale', 'fig_size', 'save_fig', 'x_label', 'y_label', 'log_scale',
            'x_range', 'y_range'.

    Returns:
        dict: The prepared dictionary of arguments for the plot.
    """
    validate_inputs(ptype=ptype, df=pd_df, x_col=x_col)
    pd_df = convert_df(pd_df)

    if pd_df.empty:
        raise ValueError("The DataFrame is empty!")
    if x_col not in pd_df.columns:
        raise ValueError(f"The column '{x_col}' is not in the DataFrame")

    return set_defaults(kwargs=kwargs, df=pd_df, x_col=x_col)


def plot_box(**kwargs: dict) -> None:
    """Create a box plot.

    Args:
        **kwargs: The keyword arguments for the box plot.
            These can include 'df', 'x_col', 'y_col', 'title', 'hue'.

    Returns:
        None
    """
    if kwargs["y_col"] is None:
        raise ValueError("The variable 'y_col' is required for a box plot")
    return sns.boxplot(
        data=kwargs["df"], x=kwargs["x_col"], y=kwargs["y_col"], hue=kwargs["hue"]
    ).set_title(kwargs["title"])


def plot_hist(**kwargs: dict) -> None:
    """Create a histogram plot.

    Args:
        **kwargs: The keyword arguments for the histogram plot.
            These can include 'df', 'x_col', 'y_col', 'title', 'hue', 'bins', 'kde', 'x_range',
            'y_range', 'log_scale'.

    Returns:
        None
    """
    kwargs["bins"] = kwargs.get("bins", "auto")
    kwargs["kde"] = kwargs.get("kde", True)
    kwargs["x_range"] = kwargs.get("x_range", (None, None))
    kwargs["y_range"] = kwargs.get("y_range", (None, None))
    kwargs["log_scale"] = kwargs.get("log_scale", (False, False))
    kwargs["y_label"] = "count"
    return sns.histplot(
        data=kwargs["df"],
        x=kwargs["x_col"],
        y=kwargs.get("y_col"),
        hue=kwargs.get("hue"),
        kde=kwargs.get("kde"),
        bins=kwargs["bins"],
        log_scale=kwargs.get("log_scale"),
    ).set_title(kwargs["title"])


def plot_scatter(**kwargs: dict) -> None:
    """Create a scatter plot.

    Args:
        **kwargs: The keyword arguments for the scatter plot.
            These can include 'df', 'x_col', 'y_col', 'title', 'hue', 'style'.

    Returns:
        None
    """
    if kwargs["y_col"] is None:
        raise ValueError("The variable 'y_col' is required for a scatter plot")
    return sns.scatterplot(
        data=kwargs["df"],
        x=kwargs["x_col"],
        y=kwargs["y_col"],
        hue=kwargs["hue"],
        style=kwargs["style"],
    ).set_title(kwargs["title"])


def plot_time_series(**kwargs: dict) -> None:
    """Create a time series plot.

    Args:
        **kwargs: The keyword arguments for the time series plot.
            These can include 'df', 'x_col', 'y_col', 'title', 'hue', 'style'.

    Returns:
        None
    """
    return sns.lineplot(
        data=kwargs["df"],
        x=kwargs["x_col"],
        y=kwargs["y_col"],
        hue=kwargs["hue"],
        style=kwargs["style"],
    ).set_title(kwargs["title"])


def plot_pair(**kwargs: dict[str, any]) -> None:
    """Create a pair plot.

    Args:
        **kwargs: The keyword arguments for the pair plot.
            These can include 'df', 'x_col', 'y_col', 'title', 'pair_plot_kind', 'pair_plot_height'.

    Returns:
        None
    """
    kwargs["pair_plot_kind"] = kwargs.get("pair_plot_kind", "reg")
    kwargs["pair_plot_height"] = kwargs.get("pair_plot_height", 10)
    ax = sns.pairplot(
        data=kwargs["df"],
        kind=kwargs["pair_plot_kind"],
        x_vars=[kwargs["x_col"]],
        y_vars=[kwargs["y_col"]],
        height=kwargs["pair_plot_height"],
    )
    ax.fig.suptitle(kwargs["title"])


def plot_bar(**kwargs: dict) -> None:
    """Create a bar plot.

    Args:
        **kwargs: The keyword arguments for the bar plot.
            These can include 'df', 'x_col', 'y_col', 'title', 'title_font_size'.

    Returns:
        None
    """
    return sns.barplot(
        data=kwargs["df"], x=kwargs["x_col"], y=kwargs["y_col"]
    ).set_title(kwargs["title"], fontsize=kwargs["title_font_size"])


def plot_count(**kwargs: dict) -> None:
    """Create a count plot.

    Args:
        **kwargs: The keyword arguments for the count plot.
            These can include 'df', 'x_col', 'hue', 'title'.

    Returns:
        None
    """
    return sns.countplot(
        data=kwargs["df"], x=kwargs["x_col"], hue=kwargs["hue"]
    ).set_title(kwargs["title"])


def plot(
    ptype: str,
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    x_col: str,
    **kwargs: dict,
) -> None:
    """Plot method for creating various types of plots.

    This is a wrapper around the seaborn plotting functions.
    It allows a more unified utilization of the seaborn plotting functions. The following plot
    types are supported:
    {"box", "hist", "scatter", "time_series", "pair", "bar", "count"}.
    See: https://seaborn.pydata.org/api.html for more information on the seaborn plotting functions.

    Args:
        ptype: (str, required) The type of plot. Can be one of
            {"box", "hist", "scatter", "time_series", "pair", "bar", "count"}
        df: (pd.DataFrame, required) The DataFrame that contains the data to plot
        x_col: (str, required) The name of the column that holds the horizontal axis data
        **kwargs:
            y_col: (str, default None) The name of the column that holds the vertical axis data.
            title: (str, default None) The title of the plot figure.
            hue: (str, default None) Grouping variable that will produce points with different
                colors. Can be either categorical or numeric, although color mapping will behave
                differently in latter case.
            style: (str, default None) Grouping variable that will produce points with
                different markers. Can have a numeric dtype but will always be treated as
                categorical.
            bins: (Union[str, int, list], default "auto") Generic bin parameter that can be the
                name of a reference rule, the number of bins, or the breaks of the bins.
                Only applies to the "hist" plot_type. Default value is "auto".
            kde: (bool, default True) If True, compute a kernel density estimate to smooth the
                distribution and show on the plot as (one or more) line(s).
                Only relevant with univariate data. Default value is True
            pair_plot_kind: (str, default "reg", accepted values {"scatter", "kde", "hist", "reg"})
                The kind of relationship plot to make
            pair_plot_height: (float, default 10) Height (in inches) of each facet
            log_scale: (tuple[bool, bool], default (False, False)) The value 'True' turns on
                logarithmic scaling for the plot axes.
                This a tuple with length two, where the first element refers to the 'x' axis,
            while the second value refers to the 'y' axis
            x_range: (tuple[float, float], default (None, None))
                Set the minimum and maximum limits of the 'x' axis
            y_range: (tuple[float, float], default (None, None))
                Set the minimum and maximum limits of the 'y' axis
            x_label: (str) The label for the X axis
            y_label: (str) The label for the Y axis
            title_font_size: (int, default 14) The size of the title font
            plot_font_scale: (int, default 1.0)  Apply a scale factor to the font size
            fig_size: (tuple[float, float], default (12, 8)) The figure size as a tuple of
                width, height in inches
            save_fig: (bool, default False)  Save the figure to file, when True. Requires file_name
                to be given
            file_name: (str, default None) The name of the file to save the figure to

    Returns:
        None: If an unrecognized plot type is specified or an error occurs.
    """
    kwargs = universal_plot_args(ptype=ptype, pd_df=df, x_col=x_col, **kwargs)
    if ptype.lower() in ("time-series", "ts", "TS"):
        ptype = "time_series"

    # Mapping plot types to their corresponding functions
    plot_func_dict = {
        "box": plot_box,
        "hist": plot_hist,
        "scatter": plot_scatter,
        "time_series": plot_time_series,
        "pair": plot_pair,
        "bar": plot_bar,
        "count": plot_count,
    }

    plot_func = plot_func_dict.get(ptype)
    if plot_func is None:
        warnings.warn(
            f"Attempting to create an unrecognized plot type: {ptype}. Skipping plot.",
            stacklevel=2,
        )
        return

    try:
        fig_plot = plot_func(**kwargs)
        if kwargs["log_scale"][0]:
            plt.xscale("log")
        if kwargs["log_scale"][1]:
            plt.yscale("log")

        plt.xlim(kwargs["x_range"][0], kwargs["x_range"][1])
        plt.ylim(kwargs["y_range"][0], kwargs["y_range"][1])

        plt.xlabel(kwargs["x_label"])
        plt.ylabel(kwargs["y_label"])
        # plt.legend(fontsize='x-large', title_fontsize=font_size)
    except ValueError as err:
        print(f"Error caught: {err}")
        return

    if kwargs.get("save_fig", False):
        file_name = kwargs.get("file_name")
        if not file_name:
            raise ValueError(
                "The value of 'file_name' is required when 'save_fig' is True"
            )
        plt.tight_layout()
        fig_plot.get_figure().savefig(kwargs["file_name"], bbox_inches="tight", dpi=300)

    # Show the plot after savefig is called
    plt.show()


def plot_corr_matrix(
    correlation_matrix: pd.DataFrame,
    is_masked: bool = False,
    font_scale: float = 1.0,
    fig_size: tuple[int, int] = (12, 12),
) -> None:
    """Plot a correlation matrix.

    Args:
        correlation_matrix (pd.DataFrame): A correlation matrix to plot
        is_masked (bool, default False): When True, mask the triangle above the diagonal
        font_scale (float, default 1.0): Apply a scale factor to the font size
        fig_size (tuple[int, int], default (12, 12)): The figure size as a tuple of width,
            height in inches

    Returns:
        None
    """
    sns.set(font_scale=font_scale)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)) if is_masked else None

    # Set up the matplotlib figure
    _ = plt.subplots(figsize=fig_size)

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        data=correlation_matrix,
        mask=mask,
        cmap="icefire",
        vmax=1,
        vmin=-1,
        center=0,
        annot=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.9},
    )


def get_numeric_cols_list(df: pd.DataFrame, drop_cols: Optional[list] = None) -> list:
    """Get a list of numeric columns from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to get the numeric columns from
        drop_cols (list, default None): A list of columns to drop from the DataFrame before
            extracting the numeric columns

    Returns:
        list: A list of numeric columns in the DataFrame
    """
    if drop_cols is None:
        drop_cols = []
    return (
        df.drop(columns=drop_cols).select_dtypes(include=[np.number]).columns.tolist()
    )


def get_categorical_cols_list(
    df: pd.DataFrame, drop_cols: Optional[list] = None
) -> list:
    """Get a list of categorical columns from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to get the categorical columns from
        drop_cols (list, default None): A list of columns to drop from the DataFrame before
            extracting the categorical columns

    Returns:
        list: A list of categorical columns in the DataFrame
    """
    if drop_cols is None:
        drop_cols = []
    return df.drop(columns=drop_cols).select_dtypes(include=["object"]).columns.tolist()
