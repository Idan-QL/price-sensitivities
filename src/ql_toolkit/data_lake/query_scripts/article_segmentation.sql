WITH analytics_data AS (
    SELECT
        uid,
        date,
        views,
        conversions,
        units_sold,
        CAST (shelf_price_stats['average'] AS REAL) AS shelf_price,
        inventory,
        CAST (cost AS REAL) AS cost,
        CAST (competitors_statistics['count'] AS INTEGER) AS competitors_count,
        CAST (competitors_statistics['min'] AS REAL) AS competitors_min_price,
        CAST (competitors_statistics['max'] AS REAL) AS competitors_max_price,
        {category_db_column_name} AS {category_df_col_name}
    FROM datalake_v1_poc.product_info_extended_daily
    WHERE client_key = '{client_key}'
      AND channel = '{channel}'
      AND date >= date('{date_str}')
),
 attributes_data AS (
     SELECT
         uid,
         date,
         CAST (element_at(element_at(attributes_map, '{elasticity_db_column_name}'), 1) AS REAL) AS {elasticity_df_col_name}
     FROM datalake_v1_poc.product_info
     WHERE client_key = '{client_key}'
       AND channel = '{channel}'
       AND date >= date('{date_str}')
 ),
 combined_data as (
     SELECT
         a.uid,
         a.date,
         a.views,
         a.conversions,
         a.units_sold,
         a.shelf_price,
         a.inventory,
         a.cost,
         a.competitors_count,
         a.competitors_min_price,
         a.competitors_max_price,
         a.{category_df_col_name},
         b.{elasticity_df_col_name},
         ROW_NUMBER() OVER (PARTITION BY a.uid, a.date ORDER BY a.date DESC) AS rn
     FROM
         analytics_data a
             LEFT JOIN
         attributes_data b
         ON
             a.uid = b.uid
                 AND a.date = b.date
     ORDER BY
         a.uid,
         a.date
     )
SELECT
    uid,
    date,
    views,
    conversions,
    units_sold,
    shelf_price,
    inventory,
    cost,
    competitors_count,
    competitors_min_price,
    competitors_max_price,
    {elasticity_df_col_name},
    {category_df_col_name}
FROM
    combined_data
WHERE
    rn = 1
ORDER BY
    uid,
    date;