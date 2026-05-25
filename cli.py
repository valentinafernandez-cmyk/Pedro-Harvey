import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import click
import httpx
from dotenv import load_dotenv

# SQLAlchemy Async Dependencies
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert


from queries.execute_query import execute_query


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
    """Inserts daily coin data and recalculates monthly aggregate stats in the db safely

    using concurrent-safe PostgreSQL upsert logic.
    """
    DailyCoinData = Base.classes.daily_coin_data
    MonthlyCoinAggregates = Base.classes.monthly_coin_aggregates

    dt = datetime.strptime(date_str, DATE_FORMAT).date()

    # 2. Safely extract price from CoinGecko payload hierarchy
    price_usd = payload.get("market_data", {}).get("current_price", {}).get("usd")

    if price_usd is None:
        return

    # 3. Daily coin data upsert
    daily_query = select(DailyCoinData).where(
        DailyCoinData.coin_id == coin_id, DailyCoinData.dt == dt
    )
    daily_record = await session.scalar(daily_query)

    if daily_record:
        daily_record.price_usd = price_usd
        daily_record.raw_payload = payload
    else:
        new_daily = DailyCoinData(
            coin_id=coin_id, dt=dt, price_usd=price_usd, raw_payload=payload
        )
        session.add(new_daily)

    # 4. Monthly coin aggregates
    yr = dt.year
    mo = dt.month
    current_utc_time = datetime.now(timezone.utc).replace(tzinfo=None)

    # Build the base PostgreSQL insertion statement
    stmt = insert(MonthlyCoinAggregates.__table__).values(
        coin_id=coin_id,
        yr=yr,
        mo=mo,
        min_price_usd=price_usd,
        max_price_usd=price_usd,
        updated_at=current_utc_time,
    )

    upsert_stmt = stmt.on_conflict_do_update(
        constraint="monthly_coin_aggregates_pkey",
        set_={
            "min_price_usd": func.least(
                MonthlyCoinAggregates.__table__.c.min_price_usd, price_usd
            ),
            "max_price_usd": func.greatest(
                MonthlyCoinAggregates.__table__.c.max_price_usd, price_usd
            ),
            "updated_at": current_utc_time,
        },
    )

    await session.execute(upsert_stmt)


async def fetch_daily_data(
    client: httpx.AsyncClient,
    coin_id: str,
    date: str,
    api_key: str,
    session_factory: Optional[sessionmaker] = None,
) -> bool:
    """Download historical coin data for a single day. Returns True if successful."""
    parse_date(date)

    try:
        params = {"date": datetime.strptime(date, DATE_FORMAT).date()}
        headers = {"x-cg-demo-api-key": api_key}

        response = await client.get(
            API_BASE_URL.format(coin_id=coin_id),
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

    except Exception as e:
        click.echo(f"\n⚠️ Error fetching {date}: {e}", err=True)
        return False

    # Routing Mechanism
    if session_factory:
        async with session_factory() as session:
            async with session.begin():
                await write_to_db(session, coin_id, date, data)
    else:
        filepath = f"{coin_id}_{date}.json"
        _save_to_json(filepath, data)
        
    return True


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

    session_factory = None

    if db_flag:
        click.echo("🔄 Connecting to database and reflecting tables...", nl=False)
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise click.ClickException(
                "\nThe --db flag was activated, but DATABASE_URL is missing from environment."
            )

        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.prepare)

        session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        click.echo(" Connected! ✅")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    # Concurrency control via bounded worker execution pool
    semaphore = asyncio.Semaphore(max_workers)
    
    # Click progress bar context manager
    with click.progressbar(
        length=len(dates_list),
        label=f"🚀 Fetching {coin_id} data",
        show_pos=True,
        fill_char="█",
        empty_char="░"
    ) as bar:

        async def worker(date_str: str, client: httpx.AsyncClient):
            async with semaphore:
                success = await fetch_daily_data(client, coin_id, date_str, api_key, session_factory)
                # Update progress bar safely across async calls using Click's standard update
                bar.update(1)
                return success

        # Dispatch concurrent HTTP operations over a unified connection pool
        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                *(worker(d, client) for d in dates_list), return_exceptions=False
            )
            
    # Clean final overview summary
    target = "Postgres DB" if db_flag else f"local JSON files in '{output_dir}'"
    click.echo(f"✨ Successfully processed and saved {len(dates_list)} days to {target}!")


@click.group()
def cli():
    """Crypto Analysis Suite CLI - Fetch data or run analysis queries."""
    pass


@cli.command(name="fetch")
@click.argument("coin-id", type=str)
@click.argument("start-date", type=str)
@click.option("--end-date", type=str, default=None, help="Optional end date for range.")
@click.option(
    "--max-workers", default=3, show_default=True, help="Max fetching workers."
)
@click.option(
    "--output-dir",
    default="./data",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=Path),
)
@click.option("--db", is_flag=True, help="Store data directly into Postgres database.")
@click.option(
    "--api-key", envvar="COINGECKO_API_KEY", required=True, help="CoinGecko API Key."
)
def fetch_coin_history(
    coin_id, start_date, end_date, max_workers, output_dir, db, api_key
):
    """Download historical coin data from CoinGecko and store it."""
    asyncio.run(
        main_async_flow(
            coin_id, start_date, end_date, max_workers, api_key, db, output_dir
        )
    )


@cli.command(name="monthly-avg")
def run_monthly_avg():
    """Calculate the historical average price for each coin grouped by month."""
    click.echo("📊 Running Monthly Average Price Analysis...")
    asyncio.run(execute_query("queries/monthly_avg.sql"))


@cli.command(name="drop-recovery")
def run_drop_recovery():
    """Analyze price recovery averages after a coin drops for 3+ consecutive days."""
    click.echo("📉 Running 3+ Day Consecutive Drop & Recovery Analysis...")
    asyncio.run(execute_query("queries/drop_recovery_analysis.sql"))


if __name__ == "__main__":
    cli()