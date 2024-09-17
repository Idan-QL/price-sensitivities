SELECT uid,
       date,
       AVG(shelf_price) AS shelf_price,
       SUM(units) AS units,
       SUM(revenue) AS revenue,
       MAX(inventory) AS inventory,
       AVG(competitor_item.price) AS avg_competitor_price
      --  (AVG(shelf_price) - AVG(competitor.price)) AS shelf_price_competitors_diff
FROM AwsDataCatalog.analytics.client_key_{client_key}
CROSS JOIN UNNEST(competitors) AS t(competitor_item)
WHERE date BETWEEN '{start_date}' AND '{end_date}'
  AND channel = '{channel}'
  {filter_units_condition}
GROUP BY uid, date;