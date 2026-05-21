import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load environment variables
load_dotenv()


def get_crypto_data(engine, days: int = 30) -> pd.DataFrame:
    """Pulls the whole table via Pandas and filters it in-memory."""

    with engine.connect() as connection:
        df = pd.read_sql_table(table_name="daily_coin_data", con=connection)

    cutoff_date = datetime.now() - timedelta(days=days)
    target_coins = ["bitcoin", "ethereum", "cardano"]

    filtered_df = df[(df["coin_id"].isin(target_coins)) & (df["dt"] >= cutoff_date)]

    sorted_df = filtered_df.sort_values(by="dt")  # type: ignore

    return sorted_df[["coin_id", "dt", "price_usd"]]


def generate_and_save_plots(df: pd.DataFrame, output_dir: str):
    """Generates and saves completely separate figure images for each asset."""
    if df.empty:
        print("⚠️ No data returned from the database for the given range.")
        return

    df["dt"] = pd.to_datetime(df["dt"])
    plot_df = df.pivot(index="dt", columns="coin_id", values="price_usd")

    crypto_styles = {
        "bitcoin": {"label": "Bitcoin (BTC)", "color": "#F7931A"},
        "ethereum": {"label": "Ethereum (ETH)", "color": "#3C3C3D"},
        "cardano": {"label": "Cardano (ADA)", "color": "#0033AD"},
    }

    os.makedirs(output_dir, exist_ok=True)

    # Loop through each coin, spin up a brand new fresh figure, and save it out
    for coin_id, style in crypto_styles.items():
        if coin_id not in plot_df.columns:
            continue

        # Initialize a dedicated canvas for *just* this coin
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            plot_df.index,
            plot_df[coin_id],
            label=style["label"],
            color=style["color"],
            linewidth=2,
        )

        # Explicit labels and titles for this separate asset context
        ax.set_xlabel("Date", fontweight="bold", labelpad=10)
        ax.set_ylabel("Price in USD", fontweight="bold", labelpad=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.title(
            f"{style['label']} Price Trend - Last 30 Days",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        fig.tight_layout()
        fig.autofmt_xdate(rotation=45)

        # Build dynamic, unique filename paths (e.g., ./plots/crypto_trend_bitcoin.png)
        specific_output_path = os.path.join(output_dir, f"crypto_trend_{coin_id}.png")
        plt.savefig(specific_output_path, dpi=300)

        # Crucial: Close the figure to free memory while running the loop
        plt.close(fig)
        print(f"📈 Chart successfully generated and saved to: {specific_output_path}")


if __name__ == "__main__":
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("❌ DATABASE_URL is missing from the environment.")

    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    engine = create_engine(db_url)

    crypto_df = get_crypto_data(engine, days=30)

    # Point directly to the directory target instead of a file
    output_directory = "./analytics/plots"
    generate_and_save_plots(crypto_df, output_directory)
