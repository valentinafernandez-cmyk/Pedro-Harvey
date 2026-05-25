import os
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
import numpy as np
import holidays

# Load environment variables
load_dotenv()


def get_risk_summary(df: pd.DataFrame) -> pd.Series:
    # ✨ FIX: Copy the dataframe immediately to prevent modifying the global df
    df = df.sort_values(by=["coin_id", "dt"]).copy()

    # 1. Calculate daily percent change (delta)
    df["delta"] = df.groupby("coin_id")["price_usd"].pct_change(1)

    # 2. Find the worst drop for each coin across any calendar month
    monthly_drops = df.groupby(["coin_id", df["dt"].dt.to_period("M")])["delta"].min()

    # 3. Find the single worst drop a coin had in ITS ENTIRE HISTORY
    overall_worst_drop = monthly_drops.groupby("coin_id").min()

    # 4. Map thresholds to a single value per coin
    def assign_tier(max_drop):
        if max_drop <= -0.50:  # 50% drop
            return "High Risk"
        elif max_drop <= -0.20:  # 20% drop
            return "Medium risk"
        return "Low risk"

    # This creates a clean Series indexed by coin_id: e.g., bitcoin -> "Low risk"
    return overall_worst_drop.apply(assign_tier).rename("risk_tier")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds new features to the coin history data with advanced temporal, 
    volumetric, and capital distribution lags.
    """
    df["dt"] = pd.to_datetime(df["dt"])

    # 1. STRUCTURAL TIMELINE GAPS CORRECTION
    df = df.set_index("dt").groupby("coin_id").resample("D").asfreq().reset_index()

    # 2. CORE TARGETS & PRICE LAG ENGINE
    df["price_lead"] = df.groupby("coin_id")["price_usd"].shift(-1)

    for lag in range(1, 8):
        df[f"price_lag_{lag}"] = df.groupby("coin_id")["price_usd"].shift(lag)

    # 3. DICTIONARY PAYLOAD EXTRACTIONS
    def safe_extract(x, field):
        if not isinstance(x, dict):
            return np.nan
        market_data = x.get("market_data", {})
        if not isinstance(market_data, dict):
            return np.nan
        return market_data.get(field, {}).get("usd", np.nan)

    df["mcap_usd"] = df["raw_payload"].apply(lambda x: safe_extract(x, "market_cap"))
    df["volume_usd"] = df["raw_payload"].apply(lambda x: safe_extract(x, "total_volume"))

    # 4. VOLUME & MARKET CAP MOMENTUM LAGS
    for lag in [1, 2, 3, 7]:
        df[f"volume_lag_{lag}"] = df.groupby("coin_id")["volume_usd"].diff(lag)
        df[f"mcap_lag_{lag}"] = df.groupby("coin_id")["mcap_usd"].diff(lag)

    # Rolling statistical metrics for volume volatility over the past week
    vol_lag_cols = [f"volume_lag_{i}" for i in [1, 2, 3]]
    df["volume_3d_std"] = df[vol_lag_cols].std(axis=1, ddof=1) #type: ignore
    df["volume_7d_trend"] =  df["volume_lag_1"] - df["volume_lag_7"]


    # 5. STATISTICAL PRICE FEATURES
    df["price_7d_trend"] = df["price_lag_1"] - df["price_lag_7"]

    lag_cols = [f"price_lag_{i}" for i in range(1, 8)]
    df["price_7d_std"] = df[lag_cols].std(axis=1, skipna=False, ddof=1)
    df["price_skew_7d"] = df[lag_cols].skew(axis=1) #type: ignore

    # 6. TRANSACTION LIQUIDITY DYNAMICS
    prev_volume = df.groupby("coin_id")["volume_usd"].shift(1)
    df["volume_velocity"] = df["volume_usd"] / (prev_volume + 1e-8)
    df["log_volume_price_interaction"] = np.log1p(df["price_usd"] * df["volume_usd"])

    # 7. TIME & CALENDAR ENGINE
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["week_of_year"] = df["dt"].dt.isocalendar().week.astype(int)

    unique_years = df["dt"].dt.year.dropna().unique().tolist()
    us_holidays = holidays.US(years=unique_years)
    cn_holidays = holidays.China(years=unique_years)

    df["is_us_holiday"] = df["dt"].apply(lambda x: int(x in us_holidays))
    df["is_china_holiday"] = df["dt"].apply(lambda x: int(x in cn_holidays))

    # 8. POST-ENGINEERING CLEANUP
    df = df.dropna(subset=["price_usd"])

    return df


if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("❌ DATABASE_URL is missing from the environment.")

    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(db_url)

    with engine.connect() as connection:
        df = pd.read_sql_table(table_name="daily_coin_data", con=connection)

    print("📋 Coin Risk Summary Dictionary:")

    # Convert the Pandas Series to a raw Python dictionary
    coin_risk_summary = get_risk_summary(df)
    risk_dict = coin_risk_summary.to_dict()

    # Use json.dumps with an indent of 4 spaces for a clean, human-readable layout
    print(json.dumps(risk_dict, indent=4))
    print("\n" + "=" * 80 + "\n")

    print("⚙️ Engineering advanced risk and mathematical trend matrices...")
    enriched_df = engineer_features(df)

    # Group by coin_id and pull the last row for each group
    sampled_df = enriched_df.groupby("coin_id").tail(3)

    columns_to_hide = [
        col
        for col in [
            "raw_payload",
            "id",
            "price_lag_7",
            "price_lag_6",
            "price_lag_5",
            "price_lag_4",
            "is_us_holiday",
            "is_china_holiday",
        ]
        if col in sampled_df.columns
    ]

    # Drop the hidden columns on the fly right before printing
    print(sampled_df.drop(columns=columns_to_hide).to_string(index=False))
