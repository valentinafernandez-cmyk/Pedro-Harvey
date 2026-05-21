import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
import click
import httpx
from dotenv import load_dotenv

# SQLAlchemy Async Dependencies
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select


load_dotenv()

API_BASE_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/history"
DATE_FORMAT = "%Y-%m-%d"

# Global Automap Container
Base = automap_base()


def _save_to_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_date(date_str: str) -> datetime:
    """Helper function to parse and validate date inputs."""
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        raise click.BadParameter("Date must be in YYYY-MM-DD format.")


async def write_to_db(
    session: AsyncSession, coin_id: str, date_str: str, payload: dict
) -> None:
    """Inserts daily coin data and recalculates monthly aggregate stats in the db."""
    DailyCoinData = Base.classes.daily_coin_data
    MonthlyCoinAggregates = Base.classes.monthly_coin_aggregates

    dt = datetime.strptime(date_str, DATE_FORMAT).date()

    # 2. Safely extract price from CoinGecko payload hierarchy
    price_usd = payload.get("market_data", {}).get("current_price", {}).get("usd")

    if price_usd is None:
        click.echo(
            f"⚠️ No USD price found in payload for {coin_id} on {date_str}.", err=True
        )
        return

    # 3. Daily coin data upsert
    # Query to see if the daily log already exists
    daily_query = select(DailyCoinData).where(
        DailyCoinData.coin_id == coin_id, DailyCoinData.dt == dt
    )
    daily_record = await session.scalar(daily_query)

    if daily_record:
        # Update existing object attributes
        daily_record.price_usd = price_usd
        daily_record.raw_payload = payload
        print("-> Tracked existing daily_coin_data for update")
    else:
        # Instantiate a new object and add it to the session
        new_daily = DailyCoinData(
            coin_id=coin_id, dt=dt, price_usd=price_usd, raw_payload=payload
        )
        session.add(new_daily)
        print("-> Added new daily_coin_data record")

    # 4. Monthly coin aggregates
    yr = dt.year
    mo = dt.month
    current_utc_time = datetime.now(timezone.utc).replace(tzinfo=None)

    monthly_query = select(MonthlyCoinAggregates).where(
        MonthlyCoinAggregates.coin_id == coin_id,
        MonthlyCoinAggregates.yr == yr,
        MonthlyCoinAggregates.mo == mo,
    )
    monthly_record = await session.scalar(monthly_query)

    if monthly_record:
        # Perform min/max check
        if price_usd < monthly_record.min_price_usd:
            monthly_record.min_price_usd = price_usd
        if price_usd > monthly_record.max_price_usd:
            monthly_record.max_price_usd = price_usd

        monthly_record.updated_at = current_utc_time
        print("-> Tracked existing monthly_coin_aggregates for update")
    else:
        # Create a brand-new aggregate record entry
        new_monthly = MonthlyCoinAggregates(
            coin_id=coin_id,
            yr=yr,
            mo=mo,
            min_price_usd=price_usd,
            max_price_usd=price_usd,
            updated_at=current_utc_time,
        )
        session.add(new_monthly)
        print("-> Added new monthly_coin_aggregates record")


async def fetch_daily_data(
    client: httpx.AsyncClient,
    coin_id: str,
    date: str,
    api_key: str,
    session_factory: Optional[sessionmaker] = None,
) -> None:
    """Download historical coin data for a single day and choose storage routing."""
    parse_date(date)

    try:
        click.echo(f"Fetching data for '{coin_id}' on {date}...")

        params = {"date": datetime.strptime(date, DATE_FORMAT).date()}
        headers = {"x-cg-demo-api-key": api_key}

        response = await client.get(
            API_BASE_URL.format(coin_id=coin_id),
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    except httpx.HTTPStatusError as e:
        click.echo(
            f"Error fetching coin data for date {date} (Status {e.response.status_code}): {e.response.text}",
            err=True,
        )
        return
    except httpx.HTTPError as e:
        click.echo(f"Network request failed for {date}: {e}", err=True)
        return
    except Exception as e:
        click.echo(f"Unexpected system error fetching {date}: {e}", err=True)
        return

    # Routing Mechanism
    if session_factory:
        # Route to PostgreSQL Target
        async with session_factory() as session:
            async with session.begin():
                await write_to_db(session, coin_id, date, data)
        click.echo(
            f"Successfully processed and committed {date} payload to Postgres DB."
        )
    else:
        # Fallback to local JSON files
        filepath = f"{coin_id}_{date}.json"
        _save_to_json(filepath, data)
        click.echo(f"Successfully saved data to {filepath}")


async def main_async_flow(
    coin_id: str,
    start_date: str,
    end_date: Optional[str],
    max_workers: int,
    api_key: str,
    db_flag: bool,
    output_dir: Path,
) -> None:
    """Orchestrates IO target initialization, schema reflection, and concurrent task dispatching."""
    start_dt = parse_date(start_date)

    if end_date is None:
        dates_list = [start_date]
        max_workers = 1
    else:
        end_dt = parse_date(end_date)
        if start_dt > end_dt:
            raise click.BadParameter("start_date must precede end-date")

        delta = end_dt - start_dt
        dates_list = [
            (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(delta.days + 1)
        ]
        click.echo(f"Processing range: {len(dates_list)} days total.")

    session_factory = None

    if db_flag:
        click.echo("Connecting to database and reflecting structural schema tables...")
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise click.ClickException(
                "The --db flag was activated, but DATABASE_URL is missing from environment."
            )

        # Initialize async database engine and execute schema reflection
        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.prepare)

        session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    else:
        # Enforce target filesystem context for local storage routing
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    # Concurrency control via bounded worker execution pool
    semaphore = asyncio.Semaphore(max_workers)

    async def worker(date_str: str, client: httpx.AsyncClient):
        async with semaphore:
            await fetch_daily_data(client, coin_id, date_str, api_key, session_factory)

    # Dispatch concurrent HTTP operations over a unified connection pool
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *(worker(d, client) for d in dates_list), return_exceptions=False
        )


@click.command()
@click.argument("coin-id", type=str)
@click.argument("start-date", type=str)
@click.option(
    "--end-date",
    type=str,
    default=None,
    help="Optional end date to fetch a range of dates.",
)
@click.option(
    "--max-workers",
    default=3,
    show_default=True,
    help="Max workers spawned when fetching a range.",
)
@click.option(
    "--output-dir",
    default="./data",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
    help="Directory to save the output files (ignored if --db is set).",
)
@click.option("--db", is_flag=True, help="Store data directly into Postgres database.")
@click.option(
    "--api-key", envvar="COINGECKO_API_KEY", required=True, help="CoinGecko API Key."
)
def cli(
    coin_id: str,
    start_date: str,
    end_date: Optional[str],
    max_workers: int,
    output_dir: Path,
    db: bool,
    api_key: str,
) -> None:
    """CLI entry point to fetch and persist historical coin data from CoinGecko."""

    # Initialize a singular async event loop lifecycle for the application runtime
    asyncio.run(
        main_async_flow(
            coin_id, start_date, end_date, max_workers, api_key, db, output_dir
        )
    )


if __name__ == "__main__":
    cli()
