-- Take the bigger units_sold source
-- SELECT 
--     date, 
--     uid,
--     CASE 
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND (product_info_analytics['units_sold'] IS NULL 
--                 OR CAST(product_events_analytics['units_sold'] AS DOUBLE) > CAST(product_info_analytics['units_sold'] AS DOUBLE))
--         THEN CAST(product_events_analytics['units_sold'] AS DOUBLE)
--         ELSE CAST(product_info_analytics['units_sold'] AS DOUBLE)
--     END AS units,
--     CASE 
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND (product_info_analytics['units_sold'] IS NULL 
--                 OR CAST(product_events_analytics['units_sold'] AS DOUBLE) > CAST(product_info_analytics['units_sold'] AS DOUBLE))
--         THEN CAST(product_events_financial['revenue'] AS DOUBLE)
--         ELSE CAST(product_info_financial['revenue'] AS DOUBLE)
--     END AS revenue,
--     CASE 
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND (product_info_analytics['units_sold'] IS NULL 
--                 OR CAST(product_events_analytics['units_sold'] AS DOUBLE) > CAST(product_info_analytics['units_sold'] AS DOUBLE))
--         THEN CAST(product_events_shelf_price_stats['most_units_sold'] AS DOUBLE)
--         ELSE CAST(product_info_shelf_price_stats['most_units_sold'] AS DOUBLE)
--     END AS shelf_price,
--     CASE 
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND product_info_analytics['units_sold'] IS NULL
--         THEN 'product_info_analytics is null, product_events_analytics taken'
--         WHEN product_events_analytics['units_sold'] IS NULL 
--             AND product_info_analytics['units_sold'] IS NOT NULL
--         THEN 'product_events_analytics is null, product_info_analytics taken'
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND CAST(product_events_analytics['units_sold'] AS DOUBLE) = CAST(product_info_analytics['units_sold'] AS DOUBLE)
--         THEN 'equal units, product_events_analytics taken'
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND (product_info_analytics['units_sold'] IS NULL 
--                 OR CAST(product_events_analytics['units_sold'] AS DOUBLE) > CAST(product_info_analytics['units_sold'] AS DOUBLE))
--         THEN 'product_events_analytics'
--         ELSE 'product_info_analytics'
--     END AS source,
--     inventory
-- FROM datalake_v1_poc.product_extended_daily
-- WHERE date BETWEEN date('{start_date}') AND date('{end_date}')
-- AND channel = '{channel}'
-- AND client_key = '{client_key}'
-- AND (
--     CASE 
--         WHEN product_events_analytics['units_sold'] IS NOT NULL 
--             AND (product_info_analytics['units_sold'] IS NULL 
--                 OR CAST(product_events_analytics['units_sold'] AS DOUBLE) > CAST(product_info_analytics['units_sold'] AS DOUBLE))
--         THEN CAST(product_events_analytics['units_sold'] AS DOUBLE)
--         ELSE CAST(product_info_analytics['units_sold'] AS DOUBLE)
--     END > 0
-- );


-- product_events_analytics is always taken unless it's NULL,
-- in which case product_info_analytics is used.
SELECT 
    date, 
    uid,
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
    CASE 
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
        THEN CAST(product_events_shelf_price_stats['most_units_sold'] AS DOUBLE)
        ELSE CAST(product_info_shelf_price_stats['most_units_sold'] AS DOUBLE)
    END AS {data_columns.shelf_price},
    CASE 
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
            AND product_info_analytics['units_sold'] IS NULL
        THEN 'product_events_analytics taken, product_info_analytics is null'
        WHEN product_events_analytics['units_sold'] IS NULL 
            AND product_info_analytics['units_sold'] IS NOT NULL
        THEN 'product_info_analytics taken, product_events_analytics is null'
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
            AND CAST(product_events_analytics['units_sold'] AS DOUBLE) > CAST(product_info_analytics['units_sold'] AS DOUBLE)
        THEN 'product_events_analytics taken, it has the bigger number'
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
            AND CAST(product_events_analytics['units_sold'] AS DOUBLE) = CAST(product_info_analytics['units_sold'] AS DOUBLE)
        THEN 'product_events_analytics taken, equal values'
        WHEN product_events_analytics['units_sold'] IS NOT NULL 
            AND CAST(product_events_analytics['units_sold'] AS DOUBLE) < CAST(product_info_analytics['units_sold'] AS DOUBLE)
        THEN 'product_events_analytics taken, but it has the smaller number'
        ELSE 'product_info_analytics taken'
    END AS source,
    inventory as {data_columns.inventory}
FROM datalake_v1_poc.product_extended_daily
WHERE date BETWEEN date('{start_date}') AND date('{end_date}')
AND channel = '{channel}'
AND client_key = '{client_key}'
  {filter_units_condition}
