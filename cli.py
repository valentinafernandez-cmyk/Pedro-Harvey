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

Base = automap_base()


def _save_to_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_date(date_str: str) -> datetime:
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
    price_usd = payload.get("market_data", {}).get("current_price", {}).get("usd")

    if price_usd is None:
        return

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

    yr = dt.year
    mo = dt.month
    current_utc_time = datetime.now(timezone.utc).replace(tzinfo=None)

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
    silent_errors: bool = False,
) -> bool:
    """Download historical coin data for a single day with built-in retry backoff for 429 errors."""
    parse_date(date)

    url = API_BASE_URL.format(coin_id=coin_id)
    params = {"date": datetime.strptime(date, DATE_FORMAT).date()}

    headers = {"x-cg-demo-api-key": api_key}

    try:
        response = await client.get(url, params=params, headers=headers, timeout=10.0)

        if response.status_code == 429:
            if not silent_errors:
                click.echo(
                    f"\n🛑 429 Status Reached (API Rate Limit) at {date}. Fetch stopped."
                )
            return False

        response.raise_for_status()
        data = response.json()

    except Exception as e:
        if not silent_errors:
            click.echo(f"\n⚠️ Failed to fetch {date}: {e}", err=True)
        return False

    # Enrutamiento a DB o JSON
    try:
        if session_factory:
            async with session_factory() as session:
                async with session.begin():
                    await write_to_db(session, coin_id, date, data)
        else:
            filepath = f"{coin_id}_{date}.json"
            _save_to_json(filepath, data)
        return True
    except Exception:
        return False


async def main_async_flow(
    coin_id: str,
    start_date: str,
    end_date: Optional[str],
    max_workers: int,
    api_key: str,
    db_flag: bool,
    missing_only: bool = False,
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
        click.echo("Connecting to database and reflecting tables...", nl=False)
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise click.ClickException(
                "\nThe --db flag was activated, but DATABASE_URL is missing."
            )

        engine = create_async_engine(db_url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.prepare)

        session_factory = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        click.echo(" Connected!")
    else:
        output_dir = Path("./data")
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)

    if missing_only and session_factory:
        DailyCoinData = Base.classes.daily_coin_data

        start_date_obj = start_dt.date()
        end_date_obj = parse_date(end_date).date() if end_date else start_date_obj

        async with session_factory() as session:
            query = select(DailyCoinData.dt).where(
                DailyCoinData.coin_id == coin_id,
                DailyCoinData.dt >= start_date_obj,
                DailyCoinData.dt <= end_date_obj,
            )
            result = await session.scalars(query)
            existing_dates = {d.strftime("%Y-%m-%d") for d in result.all()}

        original_count = len(dates_list)
        dates_list = [d for d in dates_list if d not in existing_dates]
        skipped_count = original_count - len(dates_list)

        if skipped_count > 0:
            click.echo(
                f"Found {skipped_count} days already present. Skipping those dates."
            )

        if not dates_list:
            click.echo("All specified dates already exist. Nothing to fetch!")
            return

    # Determinamos si silenciamos la consola basándonos en el volumen de días
    is_range_fetch = len(dates_list) > 1
    failed_dates = []

    semaphore = asyncio.Semaphore(max_workers)

    if is_range_fetch:
        click.echo(f" Fetching {len(dates_list)} days for {coin_id}. Please wait...")

    # Gestor de progreso condicional
    with click.progressbar(
        length=len(dates_list),
        label=f"📥 Processing {coin_id}",
        show_pos=True,
        fill_char="█",
        empty_char="░",
    ) as bar:

        async def worker(date_str: str, client: httpx.AsyncClient):
            async with semaphore:
                # Si es un rango largo, activamos el flag silent_errors para no romper la barra
                success = await fetch_daily_data(
                    client,
                    coin_id,
                    date_str,
                    api_key,
                    session_factory,
                    silent_errors=is_range_fetch,
                )
                if not success:
                    failed_dates.append(date_str)
                bar.update(1)
                return success

        async with httpx.AsyncClient() as client:
            await asyncio.gather(
                *(worker(d, client) for d in dates_list), return_exceptions=False
            )

    target = "Postgres DB" if db_flag else "local JSON files"
    successful_count = len(dates_list) - len(failed_dates)

    click.echo("\n--- Fetch Summary ---")
    click.echo(
        f"✅ Successfully processed: {successful_count}/{len(dates_list)} days saved to {target}."
    )

    if failed_dates:
        click.echo(
            f"❌ Failed to fetch {len(failed_dates)} days (likely due to API Rate Limits):"
        )
        click.echo(f"   {', '.join(sorted(failed_dates))}")


@click.group()
def cli():
    pass


@cli.command(name="fetch")
@click.argument("coin-id", type=str)
@click.argument("start-date", type=str)
@click.option("--end-date", type=str, default=None)
@click.option("--max-workers", default=3, show_default=True)
@click.option("--db", is_flag=True)
@click.option("--missing-only", is_flag=True)
@click.option("--api-key", envvar="COINGECKO_API_KEY", required=True)
def fetch_coin_history(
    coin_id, start_date, end_date, max_workers, db, missing_only, api_key
):
    """Download historical coin data from CoinGecko and store it."""
    asyncio.run(
        main_async_flow(
            coin_id, start_date, end_date, max_workers, api_key, db, missing_only
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
