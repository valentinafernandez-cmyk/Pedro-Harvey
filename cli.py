import json
import click
import httpx
from datetime import datetime
from dotenv import load_dotenv

# Automatically look for a .env file and load it into environment variables
load_dotenv()

# Coingecko (coin history) API URL
API_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/history?date={date}"

@click.command()
@click.argument('coin_id', type=str, help='Coin ID (e.g. bitcoin)')
@click.argument('date', type=str, help='Date in ISO8601 format.')
@click.option('--output-dir', default='./data', help='Directory to save the output file.')
@click.option(
    '--api-key', 
    envvar='COINGECKO_API_KEY', 
    required=True, 
    help='CoinGecko API Key. Can be set in a local .env file as COINGECKO_API_KEY.'
)
def fetch_coin_history(coin_id, date, output_dir, api_key):
    """Download historical coin data from CoinGecko.
    COIN_ID: Coin identifier (e.g., bitcoin)\n
    DATE: Date in ISO8601 format (YYYY-MM-DD)
    """

    # 1. Check date is in ISO8601 format
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter("Date must be in ISO8601 format (YYYY-MM-DD)")
    
    # 2. Make the API request with headers using httpx
    try:
        click.echo(f"Fetching data for {coin_id} on {date}...")

        response = httpx.get(
            API_URL.format(coin_id=coin_id, date=date),
            headers={"x-cg-demo-api-key": api_key}
        )
        
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as e:
        raise click.ClickException(f"API request failed: {e}")

    # 3. Save the JSON data to a file
    filename = f"{coin_id}_{date}.json"
    filepath = f"{output_dir}/{filename}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    click.echo(f"Successfully saved data to {filepath}")

if __name__ == "__main__":
    fetch_coin_history()