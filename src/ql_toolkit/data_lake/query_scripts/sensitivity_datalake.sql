SELECT uid AS {data_columns.uid},
       date AS {data_columns.date},
       CAST(COALESCE(CAST(competitors_map['googlede1'] AS ROW(price DECIMAL(15, 4), sensitivity DOUBLE)).price, NULL) AS DOUBLE) AS {data_columns.shelf_price},
       CASE
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
        THEN CAST(product_events_analytics['units_sold'] AS DOUBLE)
        ELSE CAST(product_info_analytics['units_sold'] AS DOUBLE)
        END AS {data_columns.quantity},
       CASE 
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
        THEN CAST(product_events_financial['revenue'] AS DOUBLE)
        ELSE CAST(product_info_financial['revenue'] AS DOUBLE)
        END AS {data_columns.revenue},
        inventory_stats['max'] as {data_columns.inventory}
FROM datalake_v1_poc.product_extended_daily
WHERE date BETWEEN date('{start_date}') AND date('{end_date}')
AND channel = '{channel}'
AND client_key = '{client_key}'
AND COALESCE(
        CAST(competitors_map['googlede1'] AS ROW(price DECIMAL(15, 4), sensitivity DOUBLE)).price, 
        NULL
    ) IS NOT NULL
  {filter_units_condition}