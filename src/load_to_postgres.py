import pandas as pd
from sqlalchemy import create_engine

CSV_PATH = "data/raw/paysim.csv"

DB_USER = "fraud_user"
DB_PASSWORD = "fraud_pass"
DB_HOST = "127.0.0.1"
DB_PORT = "55432"
DB_NAME = "fraud_warehouse"

TABLE_NAME = "raw_transactions"
CHUNK_SIZE = 100000

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Loading PaySim dataset into PostgreSQL...")

for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE)):
    chunk.columns = [
        "step",
        "transaction_type",
        "amount",
        "origin_customer",
        "old_balance_origin",
        "new_balance_origin",
        "destination_customer",
        "old_balance_destination",
        "new_balance_destination",
        "is_fraud",
        "is_flagged_fraud",
    ]

    if i == 0:
        chunk.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)
    else:
        chunk.to_sql(TABLE_NAME, engine, if_exists="append", index=False)

    print(f"Loaded chunk {i + 1}: {len(chunk)} rows")

print("Load completed successfully.")