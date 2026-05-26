import os
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import json
import click

def load_table_to_dataframe(table_name: str) -> pd.DataFrame:
    """Lee una tabla completa de la base de datos de Postgres y la devuelve como un DataFrame.
    """
    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError("❌ DATABASE_URL is missing from the environment.")

    # Convertimos el driver de asyncpg a psycopg2 (sincrónico) para Pandas
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    
    # Creamos el engine
    engine = create_engine(db_url)
    
    try:
        with engine.connect() as connection:
            df = pd.read_sql_table(table_name=table_name, con=connection)
        return df
    finally:
        engine.dispose()



def save_to_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def parse_date(date_str: str) -> datetime:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise click.BadParameter("Date must be in YYYY-MM-DD format.")