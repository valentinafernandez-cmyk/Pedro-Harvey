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
    """Adds new features to the coin history data with advanced scale-agnostic
    temporal, volumetric, and capital distribution ratio metrics.
    """
    df["dt"] = pd.to_datetime(df["dt"])

    # 1. STRUCTURAL TIMELINE GAPS CORRECTION
    df = df.set_index("dt").groupby("coin_id").resample("D").asfreq().reset_index()

    # 2. DICTIONARY PAYLOAD EXTRACTIONS (Moved up to support ratio anchors)
    def safe_extract(x, field):
        if not isinstance(x, dict):
            return np.nan
        market_data = x.get("market_data", {})
        if not isinstance(market_data, dict):
            return np.nan
        return market_data.get(field, {}).get("usd", np.nan)

    df["mcap_usd"] = df["raw_payload"].apply(lambda x: safe_extract(x, "market_cap"))
    df["volume_usd"] = df["raw_payload"].apply(
        lambda x: safe_extract(x, "total_volume")
    )

    # 3. CORE TARGETS & PRICE RATIO LAG ENGINE
    df["price_lead"] = df.groupby("coin_id")["price_usd"].shift(-1) / df["price_usd"]

    for lag in range(1, 8):
        raw_price_lag = df.groupby("coin_id")["price_usd"].shift(lag)
        # Ratio = Historical Price / Today's Price
        df[f"price_lag_{lag}"] = raw_price_lag / df["price_usd"]

    # 4. VOLUME & MARKET CAP MOMENTUM RATIO LAGS
    for lag in [1, 2, 3]:
        raw_vol_lag = df.groupby("coin_id")["volume_usd"].shift(lag)
        raw_mcap_lag = df.groupby("coin_id")["mcap_usd"].shift(lag)

        # Ratio = Historical Metric / Today's Metric
        df[f"volume_lag_{lag}"] = raw_vol_lag / df["volume_usd"]
        df[f"mcap_lag_{lag}"] = raw_mcap_lag / df["mcap_usd"]

    # Rolling statistical metrics for volume volatility over the past week (now on ratio scale)
    vol_lag_cols = [f"volume_lag_{i}" for i in [1, 2, 3]]
    df["volume_3d_std"] = (df[vol_lag_cols].std(axis=1, ddof=1)) / df["volume_usd"]  # type: ignore
    df["volume_3d_trend"] = (df["volume_lag_1"] - df["volume_lag_3"]) / df["volume_usd"]

    # 5. STATISTICAL PRICE FEATURES
    lag_cols = [f"price_lag_{i}" for i in range(1, 8)]

    df["price_7d_trend"] = (df["price_lag_1"] - df["price_lag_7"]) / df["price_usd"]
    df["price_7d_std"] = (
        df[lag_cols].std(axis=1, skipna=False, ddof=1) / df["price_usd"]
    )  # type: ignore
    df["price_skew_7d"] = df[lag_cols].skew(axis=1) / df["price_usd"]  # type: ignore

    # 7. TIME & CALENDAR
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
