import os
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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

    feature_cols = [
        "7d_std",
        "7d_trend",
        "price_skew_7d",
        "day_of_week",
        "is_weekend",
        "week_of_year",
        "volume_velocity",
        "log_volume_price_interaction",
        "is_us_holiday",
        "is_china_holiday",
    ]
    feature_cols += [f"price_lag_{i}" for i in range(1, 8)]
    required_cols = feature_cols + ["next_price"]

    train_data = enriched_df[train_mask].dropna(subset=required_cols).copy()
    test_data = enriched_df[test_mask].dropna(subset=required_cols).copy()

    # -------------------------------------------------------------------------
    # 2. PER-COIN FEATURE NORMALIZATION ENGINE
    # -------------------------------------------------------------------------
    print("⚖️ Normalizing feature matrices per individual asset...")
    coin_scalers = {}
    X_train_scaled = pd.DataFrame(index=train_data.index, columns=feature_cols)
    X_test_scaled = pd.DataFrame(index=test_data.index, columns=feature_cols)

    for coin_id, group in train_data.groupby("coin_id"):
        scaler = StandardScaler()
        scaler.fit(group[feature_cols])
        coin_scalers[coin_id] = scaler
        X_train_scaled.loc[group.index, feature_cols] = scaler.transform(
            group[feature_cols]
        )

    for coin_id, group in test_data.groupby("coin_id"):
        if coin_id in coin_scalers:
            scaler = coin_scalers[coin_id]
            X_test_scaled.loc[group.index, feature_cols] = scaler.transform(
                group[feature_cols]
            )
        else:
            if not hasattr(X_test_scaled, "_global_scaler"):
                global_scaler = StandardScaler().fit(train_data[feature_cols])
            X_test_scaled.loc[group.index, feature_cols] = global_scaler.transform(
                group[feature_cols]
            )

    X_train = X_train_scaled.astype(float)
    X_test = X_test_scaled.astype(float)

    # -------------------------------------------------------------------------
    # 3. DEFINE THE SCALE-AGNOSTIC TARGET RATIONALE
    # -------------------------------------------------------------------------
    y_train_ratio = train_data["next_price"] / train_data["price_lag_1"]
    y_test_ratio = test_data["next_price"] / test_data["price_lag_1"]

    # We preserve the anchors directly inside test_data for structured evaluation
    test_data["actual_next_price"] = test_data["next_price"]

    print(
        f"🚀 Training matrix ready. Train rows: {len(X_train)}, Test rows: {len(X_test)}"
    )

    # -------------------------------------------------------------------------
    # 4. BASELINE: LINEAR REGRESSION
    # -------------------------------------------------------------------------
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_ratio)

    lr_pred_ratio = lr_model.predict(X_test)
    test_data["lr_pred_usd"] = lr_pred_ratio * test_data["price_lag_1"]

    # -------------------------------------------------------------------------
    # 5. CHALLENGER: LIGHTGBM REGRESSOR (Gradient Boosting Trees)
    # -------------------------------------------------------------------------
    lgb_model = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.02,
        num_leaves=7,  # Limit tree complexity (shallow trees)
        min_child_samples=100,  # Force splits to have plenty of data points
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train_ratio)

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

    # Listas para calcular promedios globales ponderados correctamente
    all_lr_mapes = []
    all_lgb_mapes = []

    # Loop over each individual coin group in the test dataset
    for coin_id, group in test_data.groupby("coin_id"):
        y_true = group["actual_next_price"]
        y_lr = group["lr_pred_usd"]
        y_lgb = group["lgb_pred_usd"]

        # 1. Errores Absolutos en USD (MAE clásico)
        coin_lr_mae = mean_absolute_error(y_true, y_lr)
        coin_lgb_mae = mean_absolute_error(y_true, y_lgb)

        # 2. Errores Porcentuales Absolutos respecto al precio real de ese día
        # Evitamos división por cero con una pequeña constante por seguridad
        lr_percentage_errors = np.abs(y_true - y_lr) / (y_true + 1e-8) * 100
        lgb_percentage_errors = np.abs(y_true - y_lgb) / (y_true + 1e-8) * 100

        # Promedio del error porcentual para esta moneda (MAPE)
        coin_lr_mape = np.mean(lr_percentage_errors)
        coin_lgb_mape = np.mean(lgb_percentage_errors)

        # Guardamos para el resumen global
        all_lr_mapes.extend(lr_percentage_errors.tolist())
        all_lgb_mapes.extend(lgb_percentage_errors.tolist())

        # Calcular si LightGBM mejoró el error porcentual respecto a Linear Regression
        # Si el resultado es positivo, LGB redujo el error % (es mejor).
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

    print("=" * 105)
