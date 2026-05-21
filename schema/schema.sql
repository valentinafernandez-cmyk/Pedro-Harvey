-- Table 1: Daily coin data
CREATE TABLE IF NOT EXISTS daily_coin_data (
    id SERIAL PRIMARY KEY,
    coin_id VARCHAR(100) NOT NULL,
    price_usd NUMERIC(20, 8),
    dt DATE NOT NULL,
    raw_payload JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_coin_daily_snapshot UNIQUE (coin_id, dt)
);

-- Index to accelerate search by coin_id + date
CREATE INDEX IF NOT EXISTS idx_daily_coin_data ON daily_coin_data (coin_id, dt);

-- Table 2: Monthly aggregates
CREATE TABLE IF NOT EXISTS monthly_coin_aggregates (
    coin_id VARCHAR(100) NOT NULL,
    yr INT NOT NULL,
    mo INT NOT NULL,
    min_price_usd NUMERIC(20, 8) NOT NULL,
    max_price_usd NUMERIC(20, 8) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (coin_id, yr, mo)
);