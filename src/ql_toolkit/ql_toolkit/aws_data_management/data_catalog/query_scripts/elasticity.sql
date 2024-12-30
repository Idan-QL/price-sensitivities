SELECT uid as {data_columns.uid},
       date as {data_columns.date},
       AVG(shelf_price) AS {data_columns.shelf_price},
       SUM(units) AS {data_columns.quantity},
       SUM(revenue) AS {data_columns.revenue},
       MAX(inventory) AS {data_columns.inventory},
       AVG(competitor_item.price) AS {data_columns.avg_competitors_price},
       MIN(competitor_item.price) AS {data_columns.min_competitors_price},
       MAX(competitor_item.price) AS {data_columns.max_competitors_price}
FROM AwsDataCatalog.analytics.client_key_{client_key}
CROSS JOIN UNNEST(competitors) AS t(competitor_item)
WHERE date BETWEEN '{start_date}' AND '{end_date}'
  AND channel = '{channel}'
  {filter_units_condition}
GROUP BY uid, date;