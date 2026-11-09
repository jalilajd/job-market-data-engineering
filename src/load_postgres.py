from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine


def  load_data():

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "rekrute_cleaned.parquet"

    USER = "postgres"
    PASSWORD = "jalila2026"
    HOST = "host.docker.internal"
    PORT = "5433"
    DATABASE = "job_market_db"

    df = pd.read_parquet(PARQUET_PATH)

    print("Nombre d'offres :", len(df))

    engine = create_engine(
        f"postgresql+psycopg2://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
    )

    df.to_sql(
        "staging_job_offers",
        engine,
        if_exists="replace",
        index=False
    )

    print("Chargement terminé !")

    engine.dispose()