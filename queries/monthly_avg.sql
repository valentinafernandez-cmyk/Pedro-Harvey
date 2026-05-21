SELECT 
    coin_id,
    TO_CHAR(dt, 'YYYY/MM') AS year_month,
    ROUND(AVG(price_usd), 4) AS avg_price_usd
FROM 
    daily_coin_data
GROUP BY 
    coin_id, 
    TO_CHAR(dt, 'YYYY/MM')
ORDER BY 
    coin_id, 
    year_month;