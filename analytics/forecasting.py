import os
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV  # Added for hyperparameter tuning
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# IMPORT LOCAL FEATURE ENGINEERING MATRIX ENGINE
from feature_engineering import engineer_features

if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("❌ DATABASE_URL is missing from the environment.")

    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(db_url)

    with engine.connect() as connection:
        df = pd.read_sql_table(table_name="daily_coin_data", con=connection)

    print("⚙️ Engineering advanced risk and mathematical trend matrices...")
    enriched_df = engineer_features(df)

    # -------------------------------------------------------------------------
    # 1. CHRONOLOGICAL DATASET SPLIT
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

    cols_to_scale = [
        "price_7d_std",
        "price_7d_trend",
        "price_skew_7d",
        "volume_velocity",
        "log_volume_price_interaction",
        "volume_3d_std",
        "volume_7d_trend",
        "mcap_usd",
        "volume_usd"

    ]
    cols_to_scale += [f"price_lag_{i}" for i in range(1, 8)]
    cols_to_scale += [f"volume_lag_{i}" for i in [1, 2, 3, 7]]
    cols_to_scale += [f"mcap_lag_{i}" for i in [1, 2, 3, 7]] 

    passthrough_cols = [
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "is_us_holiday",
        "is_china_holiday",
    ]

    feature_cols = cols_to_scale + passthrough_cols
    required_cols = feature_cols + ["price_lead"]

    train_data = enriched_df[train_mask].dropna(subset=required_cols).copy()
    test_data = enriched_df[test_mask].dropna(subset=required_cols).copy()

    # -------------------------------------------------------------------------
    # 2. PER-COIN FEATURE NORMALIZATION
    # -------------------------------------------------------------------------
    print("⚖️ Normalizing feature matrices per individual asset...")
    
    # Structural fix: Initialize datasets with native shapes and passthrough columns
    X_train = train_data[feature_cols].copy().astype(float)
    X_test = test_data[feature_cols].copy().astype(float)
    coin_scalers = {}

    for coin_id, group in train_data.groupby("coin_id"):
        scaler = StandardScaler()
        scaler.fit(group[cols_to_scale])
        coin_scalers[coin_id] = scaler
        X_train.loc[group.index, cols_to_scale] = scaler.transform(group[cols_to_scale])

    for coin_id, group in test_data.groupby("coin_id"):
        scaler = coin_scalers[coin_id]
        X_test.loc[group.index, cols_to_scale] = scaler.transform(group[cols_to_scale])
        

    # -------------------------------------------------------------------------
    # 3. DEFINE THE SCALE-AGNOSTIC TARGET
    # -------------------------------------------------------------------------
    y_train_ratio = train_data["price_lead"] / train_data["price_lag_1"]
    y_test_ratio = test_data["price_lead"] / test_data["price_lag_1"]

    print(f"🚀 Data matrices anchored. Train rows: {len(X_train)}, Test rows: {len(X_test)}")

    # -------------------------------------------------------------------------
    # 4. BASELINE: LINEAR REGRESSION
    # -------------------------------------------------------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_ratio)

    lr_pred_ratio = lr_model.predict(X_test)
    test_data["lr_pred_usd"] = lr_pred_ratio * test_data["price_lag_1"]

    # -------------------------------------------------------------------------
    # 5. CHALLENGER: LIGHTGBM WITH TIME-SERIES GRID SEARCH
    # -------------------------------------------------------------------------
    print("🔍 Executing Chronological Grid Search for LightGBM parameters...")
    
    # TimeSeriesSplit prevents data leakage during internal cross-validation folds
    tscv = TimeSeriesSplit(n_splits=5)
    
    base_lgb = LGBMRegressor(random_state=42, verbose=-1)
    
    # Params space
    param_grid = {
        'n_estimators': [100, 150],
        'learning_rate': [0.01, 0.02, 0.005],
        'num_leaves': [7, 10, 13],
        'min_child_samples': [10, 20, 50],
    }
    
    grid_search = GridSearchCV(
        estimator=base_lgb,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train_ratio)
    
    print(f"🏆 Grid Search Optimal Parameters: {grid_search.best_params_}")
    lgb_model = grid_search.best_estimator_

    lgb_pred_ratio = lgb_model.predict(X_test)
    test_data["lgb_pred_usd"] = lgb_pred_ratio * test_data["price_lag_1"]

    # -------------------------------------------------------------------------
    # 6. GRANULAR PER-COIN PERFORMANCE COMPETITION REPORT (USD & % SCALE)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print(f"{'🏆 COIN-BY-COIN PERFORMANCE BREAKDOWN (USD & PERCENTAGE ERROR)':^105}")
    print("=" * 105)
    print(
        f"{'Coin ID':<12} | {'LR MAE ($)':<12} | {'LR MAPE (%)':<12} | {'LGB MAE ($)':<12} | {'LGB MAPE (%)':<12} | {'LGB vs LR (MAPE)'}"
    )
    print("-" * 105)

    all_lr_mapes = []
    all_lgb_mapes = []

    for coin_id, group in test_data.groupby("coin_id"):
        y_true = group["price_lead"]
        y_lr = group["lr_pred_usd"]
        y_lgb = group["lgb_pred_usd"]

        coin_lr_mae = mean_absolute_error(y_true, y_lr)
        coin_lgb_mae = mean_absolute_error(y_true, y_lgb)

        lr_percentage_errors = np.abs(y_true - y_lr) / (y_true + 1e-8) * 100
        lgb_percentage_errors = np.abs(y_true - y_lgb) / (y_true + 1e-8) * 100

        coin_lr_mape = np.mean(lr_percentage_errors)
        coin_lgb_mape = np.mean(lgb_percentage_errors)

        all_lr_mapes.extend(lr_percentage_errors.tolist())
        all_lgb_mapes.extend(lgb_percentage_errors.tolist())

        mape_improvement = (
            (coin_lr_mape - coin_lgb_mape) / (coin_lr_mape + 1e-8)
        ) * 100
        sign = "+" if mape_improvement >= 0 else ""
        icon = "🟢" if mape_improvement >= 0 else "🔴"

        print(
            f"{coin_id:<12} | "
            f"${coin_lr_mae:<10,.2f} | {coin_lr_mape:<10.2f}% | "
            f"${coin_lgb_mae:<10,.2f} | {coin_lgb_mape:<10.2f}% | "
            f"{sign}{mape_improvement:>5.1f}% {icon}"
        )

    print("-" * 105)
    
    global_lr_mae = mean_absolute_error(test_data["price_lead"], test_data["lr_pred_usd"])
    global_lgb_mae = mean_absolute_error(test_data["price_lead"], test_data["lgb_pred_usd"])
    global_lr_mape = np.mean(all_lr_mapes)
    global_lgb_mape = np.mean(all_lgb_mapes)
    global_mape_improvement = ((global_lr_mape - global_lgb_mape) / global_lr_mape) * 100
    
    print(
        f"{'GLOBAL AVG':<12} | "
        f"${global_lr_mae:<10,.2f} | {global_lr_mape:<10.2f}% | "
        f"${global_lgb_mae:<10,.2f} | {global_lgb_mape:<10.2f}% | "
        f"{'+' if global_mape_improvement >= 0 else ''}{global_mape_improvement:.1f}% {'🟢' if global_mape_improvement >= 0 else '🔴'}"
    )
    print("=" * 105)