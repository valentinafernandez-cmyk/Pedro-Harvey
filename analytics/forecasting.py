import os
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from utils import load_table_to_dataframe

# Pull in our custom technical indicators and lag setups
from feature_engineering import engineer_features


async def run_forecast(df: pd.DataFrame):
    print("⚙️ Engineering features...")
    enriched_df = engineer_features(df)
    enriched_df.columns = enriched_df.columns.astype(str)

    # -------------------------------------------------------------------------
    # 1. Split the dataset chronologically
    # -------------------------------------------------------------------------
    enriched_df = enriched_df.sort_values("dt").reset_index(drop=True)

    TRAIN_RATIO = 0.80
    min_date = enriched_df["dt"].min()
    max_date = enriched_df["dt"].max()
    total_duration = max_date - min_date
    cut_point_date = min_date + (total_duration * TRAIN_RATIO)

    print(
        f"📅 Timeline Span: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
    )
    print(
        f"🎯 Dynamic Split Threshold ({int(TRAIN_RATIO * 100)}% Oldest): {cut_point_date.strftime('%Y-%m-%d')}"
    )

    train_mask = enriched_df["dt"] < cut_point_date
    test_mask = enriched_df["dt"] >= cut_point_date

    # Group features
    general_cols = [
        "price_7d_std",
        "price_7d_trend",
        "price_skew_7d",
        "volume_3d_std",
        "volume_3d_trend",
    ]
    calendar_cols = [
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "is_us_holiday",
        "is_china_holiday",
    ]
    price_lags_cols = [f"price_lag_{i}" for i in range(1, 8)]
    volume_lags_cols = [f"volume_lag_{i}" for i in range(1, 4)]
    mcap_lags_cols = [f"mcap_lag_{i}" for i in range(1, 4)]

    # We need this master list to strip out dirty rows containing NaNs before modeling
    required_cols = list(
        set(
            general_cols
            + calendar_cols
            + price_lags_cols
            + volume_lags_cols
            + mcap_lags_cols
            + ["price_lead", "coin_id", "price_usd"]
        )
    )

    # Model 1 features
    lr_features = mcap_lags_cols + price_lags_cols

    # Model 2 features
    lgb_features = calendar_cols + general_cols

    # Drop incomplete data records and force a hard copy to dodge the SettingWithCopy warning
    train_data = enriched_df[train_mask].dropna(subset=required_cols).copy()
    test_data = enriched_df[test_mask].dropna(subset=required_cols).copy()

    # Flatten out our column blocks using array unpacking for type casting
    cat_cols = [*calendar_cols, "coin_id"]
    num_cols = [*general_cols, *price_lags_cols, *volume_lags_cols, *mcap_lags_cols]

    # Batch process type conversions natively to keep things clean and speedy
    for dataset in (train_data, test_data):
        dataset[cat_cols] = dataset[cat_cols].astype("category")
        dataset[num_cols] = dataset[num_cols].astype(float)

    X_train = train_data[required_cols]
    X_test = test_data[required_cols]

    # Extract target vector arrays
    y_train = train_data["price_lead"].values
    y_test = test_data["price_lead"].values

    # -------------------------------------------------------------------------
    # 3. Establish our primary baseline using standard Linear Regression
    # -------------------------------------------------------------------------
    print("📈 Fitting global Linear Regression baseline model...")
    X_train_lr = X_train[lr_features]
    X_test_lr = X_test[lr_features]

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train)

    lr_pred = lr_model.predict(X_test_lr)
    test_data["lr_pred"] = lr_pred

    # Keep track of training errors so LightGBM knows what's left to fix
    lr_train_pred = lr_model.predict(X_train_lr)
    train_residuals = y_train - lr_train_pred

    # -------------------------------------------------------------------------
    # 4. Spin up the LightGBM to learn from the residuals
    # -------------------------------------------------------------------------
    X_train_lgb = X_train[lgb_features]
    X_test_lgb = X_test[lgb_features]

    params = {
        "n_estimators": 100,
        "learning_rate": 0.005,
        "num_leaves": 7,
        "min_child_samples": 100,
        "random_state": 42,
        "verbose": -1,
    }

    lgb_residual_model = LGBMRegressor(**params)
    await asyncio.to_thread(lgb_residual_model.fit, X_train_lgb, train_residuals)

    # Save our clean out-of-sample prediction adjustments
    test_data["lr_residuals"] = y_test - test_data["lr_pred"].values
    test_data["lgb_pred"] = lgb_residual_model.predict(X_test_lgb)

    # -------------------------------------------------------------------------
    # 5. Build our granular, per-asset competition evaluation report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'🏆 COMPREHENSIVE TRI-MODEL BALANCED LEADERS BREAKDOWN':^95}")
    print("=" * 95)
    print(
        f"{'Coin ID':<12} | "
        f"{'LR MAPE (%)':<12} | "
        f"{'Hybrid MAPE (%)':<16} | "
        f"{'Hybrid Shift'}"
    )
    print("-" * 95)

    global_lr_errors, global_hybrid_errors = [], []

    for coin_id, group in test_data.groupby("coin_id"):
        y_true_price = group["price_lead"]
        y_lr_price = group["lr_pred"]
        pred_res = group["lgb_pred"]

        # Recompose our separate predictions back into a final hybrid result
        y_hybrid_price = y_lr_price + pred_res

        # Calculate localized percentage errors safely
        lr_pct = np.abs(y_true_price - y_lr_price) / y_true_price * 100
        hybrid_pct = np.abs(y_true_price - y_hybrid_price) / y_true_price * 100

        coin_lr_mape = np.mean(lr_pct)
        coin_hybrid_mape = np.mean(hybrid_pct)

        # Extend our global lists to track macro average updates correctly
        global_lr_errors.extend(lr_pct.tolist())
        global_hybrid_errors.extend(hybrid_pct.tolist())

        # Measure performance improvement and apply intuitive, readable status indicators
        improvement = (coin_lr_mape - coin_hybrid_mape) / coin_lr_mape * 100
        status_str = (
            f"-{improvement:>5.2f}% 🟢"
            if improvement >= 0
            else f"+{abs(improvement):>5.2f}% 🔴"
        )

        print(
            f"{coin_id:<12} | "
            f"{coin_lr_mape:<12.2f}% | "
            f"{coin_hybrid_mape:<16.2f}% | "
            f"{status_str}"
        )

    print("-" * 95)

    # Compute overall unweighted macro averages across the portfolio
    macro_lr_mape = np.mean(global_lr_errors)
    macro_hybrid_mape = np.mean(global_hybrid_errors)

    global_improvement = ((macro_lr_mape - macro_hybrid_mape) / macro_lr_mape) * 100
    global_status = (
        f"-{global_improvement:.2f}% 🟢"
        if global_improvement >= 0
        else f"+{abs(global_improvement):.2f}% 🔴"
    )

    print(
        f"{'GLOBAL AVG':<12} | "
        f"{macro_lr_mape:<12.2f}% | "
        f"{macro_hybrid_mape:<16.2f}% | "
        f"{global_status}"
    )
    print("=" * 95)


if __name__ == "__main__":
    df = load_table_to_dataframe("daily_coin_data")
    asyncio.run(run_forecast(df))
