WITH expanded_competitors AS (
    SELECT
        ped.uid,
        ped.date,
        t.key AS competitor_name,
        CAST(t.value.price AS DOUBLE) AS competitor_price,
        CASE 
            WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                THEN CAST(ped.product_events_shelf_price_stats['most_units_sold'] AS DOUBLE)
            ELSE 
                CAST(ped.product_info_shelf_price_stats['most_units_sold'] AS DOUBLE) 
        END AS shelf_price_base,
        CASE 
            WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                THEN CAST(ped.product_events_analytics['units_sold'] AS DOUBLE)
            ELSE 
                CAST(ped.product_info_analytics['units_sold'] AS DOUBLE)
        END AS quantity,
        0 AS revenue,
        ped.inventory_stats['max'] AS inventory,
        -- Calculate total units sold per uid across all dates
        SUM(
            CASE 
                WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                    THEN CAST(ped.product_events_analytics['units_sold'] AS DOUBLE)
                ELSE 
                    CAST(ped.product_info_analytics['units_sold'] AS DOUBLE)
            END
        ) OVER (PARTITION BY ped.uid) AS total_units_sold
    FROM
        datalake_v1_poc.product_extended_daily ped
    CROSS JOIN UNNEST(ped.competitors_map) AS t(key, value)
    WHERE
        ped.date BETWEEN DATE('2024-05-01') AND DATE('2024-10-31')
        AND ped.channel = '{channel}'
        AND ped.client_key = '{client_key}'
        AND ped.uid = '{uid}'
)

SELECT
    ec.uid,
    ec.date,
    ec.total_units_sold,
    ec.competitor_name,
    ec.shelf_price_base,
    ec.competitor_price,
    ec.shelf_price_base - ec.competitor_price AS diff_shelf_price_minus_competitor_price,
    ec.shelf_price_base / ec.competitor_price AS ratio_shelf_price_competitor,
    ec.quantity,
    ec.revenue,
    ec.inventory
FROM 
    expanded_competitors ec
WHERE
    ec.competitor_price IS NOT NULL
    AND ec.inventory > 0;