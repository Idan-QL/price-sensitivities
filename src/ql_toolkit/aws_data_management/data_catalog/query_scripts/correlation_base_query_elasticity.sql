WITH elasticity_details AS (
    SELECT
        uid,
        attributes_map['qlia_elasticity_model'][1] AS elasticity_model,
        attributes_map['qlia_elasticity_calc_date'][1] AS elasticity_calc_date,
        attributes_map['qlia_elasticity_details'][1] AS elasticity_details
    FROM (
        SELECT
            ped.*,
            ROW_NUMBER() OVER (PARTITION BY uid ORDER BY date DESC) AS rn
        FROM
            datalake_v1_poc.product_extended_daily ped
        WHERE
            attributes_map['qlia_elasticity_model'] IS NOT NULL
            AND date BETWEEN DATE('2024-11-03') AND DATE('2024-11-09')
            AND channel = '{channel}'
            AND client_key = '{client_key}'
            AND attributes_map['qlia_elasticity_calc_date'][1] LIKE '2024-08-%'
            AND attributes_map['qlia_elasticity_details'][1] = 'High quality test.'
    ) ped
    WHERE rn = 1
),
expanded_competitors AS (
    SELECT
        ped.uid,
        ped.date,
        ed.elasticity_model,
        ed.elasticity_calc_date,
        ed.elasticity_details,
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
    INNER JOIN
        elasticity_details ed
    ON
        ped.uid = ed.uid
    CROSS JOIN UNNEST(ped.competitors_map) AS t(key, value)
    WHERE
        ped.date BETWEEN DATE('2023-11-01') AND DATE('2024-10-31')
        AND ped.channel = '{channel}'
        AND ped.client_key = '{client_key}'
)
SELECT
    uid,
    date,
    elasticity_model,
    elasticity_calc_date,
    elasticity_details,
    competitor_name,
    shelf_price_base,
    competitor_price,
    shelf_price_base - competitor_price as diff_shelf_price_minus_competitor_price,
    shelf_price_base / competitor_price AS ratio_shelf_price_competitor,
    quantity,
    revenue,
    inventory
FROM expanded_competitors
WHERE competitor_price IS NOT NULL
AND inventory > 0;
