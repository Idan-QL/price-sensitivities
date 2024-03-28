"""Module of preprocessing."""
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from elasticity.data.utils import (
    preprocess_by_price,
    round_price_effect,
    uid_with_min_conversions,
    uid_with_price_changes,
)
from ql_toolkit.config.runtime_config import app_state


def read_and_preprocess(client_key: str,
                        channel: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        bucket: Optional[str] = None,
                        dir_: str = 'data_science/datasets',
                        price_changes: int = 4, threshold: float = 0.01,
                        min_days_with_conversions: int = 15,
                        uid_col: str = 'uid',
                        price_col: str = 'round_price',
                        quantity_col: str = 'units',
                        date_col: str = 'date') ->tuple[pd.DataFrame, pd.DataFrame]:
    """Read and preprocess the DataFrame.

    TODO: add original price_col base for round_price.

    Returns:
    - df_by_day (DataFrame): DataFrame grouped by day.
    """
    if end_date is None:
        end_date = datetime.now().replace(day=1) - relativedelta(months=1)
        end_date = end_date.strftime('%Y-%m-%d')

    if start_date is None:
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_date_dt = end_date_dt - relativedelta(months=11)
        start_date = start_date_dt.strftime('%Y-%m-%d')

    if bucket is None:
        bucket = app_state.bucket_name

    logging.info("start_date: %s", start_date)
    logging.info("end_date: %s", end_date)

    df, total_end_date_uid = read_monthly_data(client_key=client_key,
                           channel=channel,
                           bucket=bucket,
                           start_date=start_date,
                           end_date=end_date,
                           dir_=dir_,
                           price_changes=price_changes, threshold=threshold,
                           min_days_with_conversions=min_days_with_conversions,
                           uid_col=uid_col,
                           price_col=price_col,
                           quantity_col=quantity_col)

    df_by_price = preprocess_by_price(df,
                                      uid_col=uid_col,
                                      date_col=date_col,
                                      price_col=price_col,
                                      quantity_col=quantity_col)

    return df_by_price, total_end_date_uid, end_date



def read_monthly_data(client_key: str, channel: str, bucket: str,
                      start_date: str, end_date: str,
                      dir_: str = 'data_science/datasets',
                      price_changes: int = 4, threshold: float = 0.01,
                      min_days_with_conversions: int = 15,
                      uid_col: str = 'uid',
                      price_col: str = 'round_price',
                      quantity_col: str = 'units'
                      ) -> pd.DataFrame:
    """Read monthly data and concatenate into a single DataFrame."""
    df_full_list = []
    uid_ok = []
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    df_ko = pd.DataFrame()  # Initialize df_ko
    total_end_date_uid = 0
    while end_date_dt >= start_date_dt:
        logging.info("reading: %s", end_date_dt.strftime('%Y-%m-%d'))
        df_part = read_data(client_key,
                            channel,
                            bucket,
                            date=end_date_dt.strftime('%Y-%m-%d'), dir_=dir_)
        if total_end_date_uid ==0:
            total_end_date_uid = df_part['uid'].nunique()
            logging.info("Total uid: %s", total_end_date_uid)
        df_part = df_part[~df_part['uid'].isin(uid_ok)]

        # Concatenate df_part with df_ko
        df_part = pd.concat([df_part, df_ko])

        uid_changes = uid_with_price_changes(df_part,
                                             price_changes=price_changes,
                                             threshold=threshold,
                                             price_col=price_col,
                                             quantity_col=quantity_col)
        uid_conversions = uid_with_min_conversions(df_part,
                                                   min_days_with_conversions=min_days_with_conversions,
                                                   uid_col=uid_col,
                                                   quantity_col=quantity_col)
        uid_intersection_change_conversions = list(set(uid_changes) & set(uid_conversions))
        uid_ok.extend(uid_intersection_change_conversions)

        df_ok = df_part[df_part['uid'].isin(uid_intersection_change_conversions)]
        logging.info("number of uid ok: %s", len(uid_ok))
        df_ko = df_part[~df_part['uid'].isin(uid_intersection_change_conversions)]

        df_full_list.append(df_ok)
        end_date_dt -= pd.DateOffset(months=1)

    result_df = pd.concat(df_full_list)
    logging.info('Number of unique user IDs: %s', result_df["uid"].nunique())
    return result_df, total_end_date_uid



def read_data(client_key: str, channel: str, bucket: str,
                      date: str = '2023-02-01',
                      dir_: str = 'data_science/datasets') -> pd.DataFrame:
    """Read monthly data and concatenate into a single DataFrame."""
    year_, month_ = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m').split("-")
    cs = ['date',
        'uid',
        'conversions_most_common_shelf_price',
        'views_most_common_shelf_price',
        'total_units',
        'price',
        'inventory'
        ]
    try:
        df_read = pd.read_parquet(f's3://{bucket}/{dir_}/{client_key}/{channel}/elasticity/{year_}_{int(month_)}_full_data.parquet/',
                                  columns=cs)
        # delete uid where inventory is 0
        logging.info("Number of inventory less or equal to 0: %s",
                     len(df_read[df_read['inventory'] <= 0]))
        df_read = df_read[df_read['inventory'] > 0]
        del df_read['inventory']
        df_read = process_data(df_read)
    except Exception:
        logging.info('No data for %s_%s}', (str(year_), str(int(month_))))
        logging.info('s3://%s/%s/%s/%s/elasticity/%s_%s_full_data.parquet/',
                     (bucket, dir_, client_key, channel, year_, int(month_)))
        df_read = pd.DataFrame(columns=cs)
        pass
    return df_read


def process_data(df_full: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional columns based on the read monthly data."""
    df_full['price_merged'] = np.where(~df_full.conversions_most_common_shelf_price.isna(),
                                       df_full.conversions_most_common_shelf_price,
                                       np.where(~df_full.views_most_common_shelf_price.isna(),
                                                df_full.views_most_common_shelf_price,
                                                df_full.price))

    df_full['source'] = np.where(~df_full.conversions_most_common_shelf_price.isna(),
                                 'conversions',
                                 np.where(~df_full.views_most_common_shelf_price.isna(),
                                          'views',
                                          'price_recommendations'))

    df_full['units'] = np.where(~df_full.total_units.isna(), df_full.total_units, 0)
    df_full['round_price'] = df_full['price_merged'].apply(round_price_effect)

    return df_full[['date','uid','round_price','units','price_merged','source']]

