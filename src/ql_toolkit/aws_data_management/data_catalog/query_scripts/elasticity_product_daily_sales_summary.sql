SELECT uid,
       date,
       CAST(avg_transaction_price_incl_vat AS DOUBLE) AS {data_columns.shelf_price},
       CAST(total_units_sold_per_day AS INT) AS {data_columns.quantity},
       CAST(daily_revenue_incl_vat AS DOUBLE) AS {data_columns.revenue},
       10 as inventory
FROM datalake_v1_poc.product_daily_sales_summary
WHERE date BETWEEN date('{start_date}') AND date('{end_date}')
  AND channel = '{channel}'
  AND client_key = '{client_key}'
  {filter_units_condition}