import os
import click
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text


# Helper to execute raw SQL files
async def execute_query(file_path: str):
    """Reads a local .sql file and executes it against the Postgres database."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise click.ClickException(
            "DATABASE_URL is missing from environment variables."
        )

    # Simple check to make sure you saved the file in the right spot
    if not os.path.exists(file_path):
        raise click.ClickException(f"SQL file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        query_text = f.read()

    engine = create_async_engine(db_url)
    async with engine.connect() as conn:
        result = await conn.execute(text(query_text))
        # Fetch all results and column keys for formatting
        rows = result.fetchall()
        keys = result.keys()

        if not rows:
            click.echo("No data found for this analysis.")
            return

        # Print a simple clean text table to the console
        header = " | ".join(keys)
        click.echo(header)
        click.echo("-" * len(header))
        for row in rows:
            click.echo(" | ".join(str(val) for val in row))
