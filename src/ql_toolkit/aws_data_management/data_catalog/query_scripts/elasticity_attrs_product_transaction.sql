SELECT uid, {attr_columns}
FROM (
    SELECT *,
        ROW_NUMBER() OVER (PARTITION BY client_key, channel, uid ORDER BY date DESC) as rn
    FROM datalake_v1_poc.product_info_daily
    WHERE client_key = '{client_key}' AND channel = '{channel}' AND date BETWEEN date '{start_date}' AND date '{end_date}'
) pid
WHERE rn = 1