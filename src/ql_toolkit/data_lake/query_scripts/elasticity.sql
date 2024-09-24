SELECT uid as {data_columns.uid},
       date as {data_columns.date},
       AVG(shelf_price) AS {data_columns.shelf_price},
       SUM(units) AS {data_columns.quantity},
       SUM(revenue) AS {data_columns.revenue},
       MAX(inventory) AS {data_columns.inventory}
FROM AwsDataCatalog.analytics.client_key_{client_key}
WHERE date BETWEEN '{start_date}' AND '{end_date}'
  AND channel = '{channel}'
  {filter_units_condition}
GROUP BY uid, date;