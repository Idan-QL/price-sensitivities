WITH expanded_competitors AS (
    SELECT
        ped.uid,
        ped.date,
        t.key AS competitor_name,
        CAST(t.value.price AS DOUBLE) AS competitor_price,
        (
            CASE 
                WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                THEN CAST(ped.product_events_shelf_price_stats['most_units_sold'] AS DOUBLE)
                ELSE CAST(ped.product_info_shelf_price_stats['most_units_sold'] AS DOUBLE) 
            END
        ) AS shelf_price_base,
        (
            CASE 
                WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                THEN CAST(ped.product_events_analytics['units_sold'] AS DOUBLE)
                ELSE CAST(ped.product_info_analytics['units_sold'] AS DOUBLE)
            END
        ) AS quantity,
        0 AS revenue,
        ped.inventory_stats['max'] AS inventory
    FROM
        datalake_v1_poc.product_extended_daily ped
    CROSS JOIN UNNEST(ped.competitors_map) AS t(key, value)
    WHERE
        ped.date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
        AND ped.channel = '{channel}'
        AND ped.client_key = '{client_key}'
)
SELECT
    uid,
    date,
    competitor_name,
    shelf_price_base,
    competitor_price,
    shelf_price_base - competitor_price AS diff_shelf_price_minus_competitor_price,
    shelf_price_base / competitor_price AS ratio_shelf_price_competitor,
    quantity,
    revenue,
    inventory
FROM expanded_competitors
WHERE competitor_price IS NOT NULL
AND inventory > 0;
