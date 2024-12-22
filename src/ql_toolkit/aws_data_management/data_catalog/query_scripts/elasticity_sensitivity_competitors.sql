WITH expanded_competitors AS (
    SELECT
        ped.uid,
        ped.date,
        t.key AS competitor_name,
        CAST(t.value.price AS DOUBLE) AS competitor_price,
        CASE 
            WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                THEN CAST(ped.product_events_shelf_price_stats['most_units_sold'] AS DOUBLE)
            ELSE CAST(ped.product_info_shelf_price_stats['most_units_sold'] AS DOUBLE) 
        END AS shelf_price_base,
        CASE 
            WHEN ped.product_events_analytics['units_sold'] IS NOT NULL 
                THEN CAST(ped.product_events_analytics['units_sold'] AS DOUBLE)
            ELSE CAST(ped.product_info_analytics['units_sold'] AS DOUBLE)
        END AS quantity,
        0 AS revenue,
        ped.inventory_stats['max'] AS inventory
    FROM
        datalake_v1_poc.product_extended_daily ped
    CROSS JOIN UNNEST(ped.competitors_map) AS t(key, value)
    WHERE
        ped.date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
        AND ped.channel = '{channel}'
        AND ped.client_key = '{client_key}'
),
final_data AS (
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
      AND inventory > 0
),
date_range AS (
    SELECT COUNT(DISTINCT date) AS total_date_count 
    FROM final_data
),
total_uids AS (
    SELECT COUNT(DISTINCT uid) AS total_uid_count 
    FROM final_data
),
competitor_uid_coverage AS (
    SELECT
        competitor_name,
        uid,
        COUNT(DISTINCT date) AS competitor_uid_date_count
    FROM final_data
    GROUP BY competitor_name, uid
),
competitor_80pct_coverage_uids AS (
    SELECT
        cuc.competitor_name,
        COUNT(*) AS uids_with_80pct_coverage
    FROM competitor_uid_coverage cuc
    CROSS JOIN date_range dr
    WHERE (competitor_uid_date_count * 1.0 / total_date_count) >= 0.8
    GROUP BY competitor_name
),
competitor_meets_threshold AS (
    SELECT
        c80.competitor_name
    FROM competitor_80pct_coverage_uids c80
    CROSS JOIN total_uids tu
    WHERE (uids_with_80pct_coverage * 1.0 / total_uid_count) >= 0.25
)
SELECT competitor_name
FROM competitor_meets_threshold
WHERE SUBSTRING(competitor_name, LENGTH(competitor_name), 1) IN ('1', '2')
   OR SUBSTRING(competitor_name, LENGTH(competitor_name), 1) BETWEEN 'a' AND 'z'
   OR SUBSTRING(competitor_name, LENGTH(competitor_name), 1) BETWEEN 'A' AND 'Z';
