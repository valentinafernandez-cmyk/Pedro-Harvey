import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import click
import httpx
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/history"

DATE_FORMAT = "%Y-%m-%d"


def _save_to_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_date(date_str: str) -> datetime:
    """Helper function to parse and validate date inputs."""
    try:
        return datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        raise click.BadParameter("Date must be in YYYY-MM-DD format.")


async def fetch_daily_data(
    client: httpx.AsyncClient,
    coin_id: str,
    date: str,
    output_dir: Path,
    api_key: str,
) -> None:
    """Download historical coin data for a single day and save it to disk."""

    # Note: validation happens prior to task spawning, but serves as a quick runtime sanity check here
    parse_date(date)

    try:
        click.echo(f"Fetching data for '{coin_id}' on {date}...")
        params = {"date": date}
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

    filepath = output_dir / f"{coin_id}_{date}.json"

    # Writing is a blocking operation, we could hand it to a background thread.
    # However, we are dealing with small files and low concurrency, so its not a bottleneck.
    # await asyncio.to_thread(_save_to_json, filepath, data)
    _save_to_json(filepath, data)
    click.echo(f"Successfully saved data to {filepath}")


async def orchestrate_tasks(
    coin_id: str,
    dates_list: List[str],
    max_workers: int,
    output_dir: Path,
    api_key: str,
) -> None:
    """Orchestrates async workers to process target dates concurrently."""
    semaphore = asyncio.Semaphore(max_workers)

    async def worker(date_str: str, client: httpx.AsyncClient):
        async with semaphore:
            await fetch_daily_data(client, coin_id, date_str, output_dir, api_key)

    async with httpx.AsyncClient() as client:
        tasks = [worker(d, client) for d in dates_list]
        await asyncio.gather(*tasks, return_exceptions=True)


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
    help="Directory to save the output files.",
)
@click.option(
    "--api-key",
    envvar="COINGECKO_API_KEY",
    required=True,
    help="CoinGecko API Key. Falls back to COINGECKO_API_KEY env var.",
)
def cli(
    coin_id: str,
    start_date: str,
    end_date: Optional[str],
    max_workers: int,
    output_dir: Path,
    api_key: str,
) -> None:
    """Download historical coin data from CoinGecko for a single date or a range."""
    # 1. Parse and validate start date
    start_dt = parse_date(start_date)

    # 2. Determine target dates list
    if end_date is None:
        # Single day execution
        dates_list = [start_date]
        max_workers = 1  # No need for max_workers for 1 file
    else:
        # Range execution
        end_dt = parse_date(end_date)

        if start_dt > end_dt:
            raise click.BadParameter("start_date must precede end-date")

        delta = end_dt - start_dt
        dates_list = [
            (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(delta.days + 1)
        ]
        click.echo(f"Processing range: {len(dates_list)} days total.")

    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        orchestrate_tasks(
            coin_id=coin_id,
            dates_list=dates_list,
            max_workers=max_workers,
            output_dir=output_dir,
            api_key=api_key,
        )
    )


if __name__ == "__main__":
    cli()
