SELECT uid AS {data_columns.uid},
       date AS {data_columns.date},
       AVG(
           CASE 
               WHEN shelf_price BETWEEN -1000 AND -998 THEN ROUND(revenue / NULLIF(units, 0), 2)
               ELSE shelf_price
           END
       ) AS {data_columns.shelf_price},
       SUM(units) AS {data_columns.quantity},
       SUM(revenue) AS {data_columns.revenue},
       MAX(inventory) AS {data_columns.inventory}
FROM AwsDataCatalog.analytics.client_key_{client_key}
WHERE date BETWEEN '{start_date}' AND '{end_date}'
  AND channel = '{channel}'
  {filter_units_condition}
GROUP BY uid, date;