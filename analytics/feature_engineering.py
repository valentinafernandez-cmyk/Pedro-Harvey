import os
import json
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Transforms crypto data according to requirements.

    Returns:
        - A DataFrame with Part B features (7d trend & variance per row)
        - A Series with Part A features (Overall risk tier per coin)
    """
    # Force chronological order per coin for safe time-series windowing
    df = df.sort_values(by=["coin_id", "dt"]).copy()
    df["dt"] = pd.to_datetime(df["dt"])

    # -------------------------------------------------------------------------
    # PART A: OVERALL RISK CLASSIFICATION PER COIN
    # -------------------------------------------------------------------------
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
    coin_risk_summary = overall_worst_drop.apply(assign_tier).rename("risk_tier")

    # -------------------------------------------------------------------------
    # PART B: 7-DAY HISTORICAL TREND AND VARIANCE (ADDED PER ROW-DAY)
    # -------------------------------------------------------------------------
    # Create a temporary column shifted by 1 day to ensure we only look at history (T-1 backwards)
    t_minus_1 = df.groupby("coin_id")["price_usd"].shift(1)

    # Calculate 7-day variance
    df["7d_var"] = (
        t_minus_1.groupby(df["coin_id"])
        .rolling(window=7, min_periods=7)
        .var()
        .reset_index(level=0, drop=True)
    )

    # Calculate trend: Price at T-1 minus Price at T-7
    t_minus_7 = df.groupby("coin_id")["price_usd"].shift(7)
    df["7d_trend"] = t_minus_1 - t_minus_7

    # Clean up temporary calculation columns from the main dataframe
    df = df.drop(columns=["delta"])

    # Return BOTH pieces of data cleanly
    return df, coin_risk_summary


if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("❌ DATABASE_URL is missing from the environment.")

    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(db_url)

    with engine.connect() as connection:
        df = pd.read_sql_table(table_name="daily_coin_data", con=connection)

    enriched_df, coin_risk_summary = engineer_features(df)

    print("📋 Coin Risk Summary Dictionary:")

    # Convert the Pandas Series to a raw Python dictionary
    risk_dict = coin_risk_summary.to_dict()

    # Use json.dumps with an indent of 4 spaces for a clean, human-readable layout
    print(json.dumps(risk_dict, indent=4))
    print("\n" + "=" * 80 + "\n")

    # Group by coin_id and pull the last 5 records for each group
    sampled_df = enriched_df.groupby("coin_id").tail(5)

    # Pretty-print the subset table cleanly
    columns_to_show = ["coin_id", "dt", "price_usd", "7d_var", "7d_trend"]
    print(sampled_df[columns_to_show].to_string(index=False))
