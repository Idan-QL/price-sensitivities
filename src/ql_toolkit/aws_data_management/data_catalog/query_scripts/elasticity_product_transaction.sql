SELECT uid AS {data_columns.uid},
       DATE(transaction_timestamp) AS {data_columns.date},
       CAST(AVG(transaction_price_incl_vat) AS REAL) AS {data_columns.shelf_price},
       CAST(SUM(units_sold) AS INT) AS {data_columns.quantity},
       CAST(SUM(revenue) as REAL) AS {data_columns.revenue},
       10 AS {data_columns.inventory}
FROM datalake_v1_poc.product_transaction
WHERE DATE(transaction_timestamp) BETWEEN date '{start_date}' AND date '{end_date}'
    AND client_key = '{client_key}'
    AND channel = '{channel}'
    {filter_units_condition}
GROUP BY uid, DATE(transaction_timestamp);