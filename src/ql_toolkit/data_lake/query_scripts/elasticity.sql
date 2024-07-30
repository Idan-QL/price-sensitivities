SELECT uid,
       date,
       AVG(shelf_price) AS shelf_price,
       SUM(units) AS units,
       SUM(revenue) AS revenue,
       MAX(inventory) AS inventory
FROM AwsDataCatalog.analytics.client_key_{client_key}
WHERE date BETWEEN '{start_date}' AND '{end_date}'
  AND channel = '{channel}'
  {filter_units_condition}
GROUP BY uid, date;