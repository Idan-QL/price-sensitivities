WITH ranked_attrs AS (
    SELECT
        uid,
        date,
        t.element.name AS attr_name,
        t.element.value AS attr_value,
        ROW_NUMBER() OVER (PARTITION BY uid, t.element.name ORDER BY date DESC) AS rn
    FROM AwsDataCatalog.analytics.client_key_{client_key}, UNNEST(attrs) AS t(element)
    WHERE t.element.name IN ({attr_names_str})
        AND (t.element.value IS NOT NULL AND t.element.value != '')
        AND date BETWEEN '{start_date}' AND '{end_date}'
        AND channel = '{channel}'
)
SELECT uid, date, attr_name, attr_value
FROM ranked_attrs
WHERE rn = 1;