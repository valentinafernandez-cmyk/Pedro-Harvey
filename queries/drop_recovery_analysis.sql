WITH consecutive_dates AS (
    -- Step 1: Group rows by asset into continuous calendar stretches
    SELECT 
        coin_id, dt, price_usd,
        (dt - ROW_NUMBER() OVER (PARTITION BY coin_id ORDER BY dt)::INT) AS date_island_id
    FROM daily_coin_data
),
market_metrics AS (
    -- Step 2: Compute relative drops and lookahead recovery metrics upfront
    SELECT 
        *,
        CASE WHEN price_usd < LAG(price_usd) OVER w THEN 1 ELSE 0 END AS is_drop,
        CASE WHEN price_usd < LAG(price_usd) OVER w 
              AND price_usd < LEAD(price_usd) OVER w 
             THEN 1 ELSE 0 END AS is_local_minima,
        (LEAD(price_usd) OVER w - price_usd) / price_usd AS next_day_delta
    FROM consecutive_dates
    WINDOW w AS (PARTITION BY coin_id, date_island_id ORDER BY dt)
),
drop_streaks AS (
    -- Step 3: Bundle consecutive drops into unique, isolated sequence buckets
    SELECT 
        *,
        CASE 
            WHEN is_drop = 1 THEN
                SUM(CASE WHEN is_drop = 0 THEN 1 ELSE 0 END) OVER (
                    PARTITION BY coin_id, date_island_id ORDER BY dt 
                )
            ELSE NULL 
        END AS streak_id
    FROM market_metrics
),
qualified_streaks AS (
    -- Step 4: pinpoints the exact next_day_delta of the local minimum for qualifying streaks.
    SELECT 
        coin_id,
        -- Pull the recovery delta exactly from the row that represents the bottom element
        MAX(next_day_delta) FILTER (WHERE is_local_minima = 1) AS bottom_recovery_delta
    FROM drop_streaks
    WHERE streak_id IS NOT NULL
    GROUP BY coin_id, date_island_id, streak_id
    HAVING COUNT(*) >= 3            -- Rule 1: Streak group must contain 3 or more days
       AND SUM(is_local_minima) = 1 -- Rule 2: The last element of the streak must be a local minima
),
latest_market_cap AS (
    -- Step 6: Isolate the single most recent market cap profile per coin
    SELECT DISTINCT ON (coin_id)
        coin_id,
        (raw_payload -> 'market_data' -> 'market_cap' ->> 'usd')::NUMERIC AS current_market_cap_usd
    FROM daily_coin_data
    ORDER BY coin_id, dt DESC
)
-- Final Select: Smoothly average the pre-isolated bottom recovery metrics at the coin level
SELECT 
    qs.coin_id,
    lmc.current_market_cap_usd,
    COUNT(*) AS total_streaks_found,
    ROUND(AVG(qs.bottom_recovery_delta) * 100, 2) || '%' AS avg_recovery_percentage_after_drop
FROM qualified_streaks qs
LEFT JOIN latest_market_cap lmc ON qs.coin_id = lmc.coin_id
GROUP BY qs.coin_id, lmc.current_market_cap_usd
ORDER BY lmc.current_market_cap_usd DESC NULLS LAST;