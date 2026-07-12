import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

DB_USER = "fraud_user"
DB_PASSWORD = "fraud_pass"
DB_HOST = "127.0.0.1"
DB_PORT = "55432"
DB_NAME = "fraud_warehouse"

OUTPUT_DIR = Path("rag_assistant/data/case_summaries")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

query = """
select
    transaction_key,
    step,
    transaction_type,
    amount,
    old_balance_origin,
    new_balance_origin,
    old_balance_destination,
    new_balance_destination,
    origin_balance_change,
    destination_balance_change,
    transaction_to_balance_ratio,
    high_risk_transaction_type,
    high_amount_flag,
    emptied_origin_account_flag,
    suspicious_transfer_flag,
    is_fraud
from fraud_features
where is_fraud = 1
limit 1000
"""

print("Reading high-risk fraud cases from PostgreSQL...")
df = pd.read_sql(query, engine)

print(f"Generating {len(df)} fraud case summaries...")

for _, row in df.iterrows():
    summary = f"""
Fraud Case ID: {row['transaction_key']}

Transaction Overview:
- Step: {row['step']}
- Transaction Type: {row['transaction_type']}
- Amount: {row['amount']}
- Actual Fraud Label: {row['is_fraud']}

Account Balance Behavior:
- Old Origin Balance: {row['old_balance_origin']}
- New Origin Balance: {row['new_balance_origin']}
- Origin Balance Change: {row['origin_balance_change']}
- Old Destination Balance: {row['old_balance_destination']}
- New Destination Balance: {row['new_balance_destination']}
- Destination Balance Change: {row['destination_balance_change']}

Risk Indicators:
- Transaction to Balance Ratio: {row['transaction_to_balance_ratio']}
- High Risk Transaction Type: {row['high_risk_transaction_type']}
- High Amount Flag: {row['high_amount_flag']}
- Emptied Origin Account Flag: {row['emptied_origin_account_flag']}
- Suspicious Transfer Flag: {row['suspicious_transfer_flag']}

Investigator Summary:
This transaction was labeled as fraudulent. The risk profile is based on transaction amount, balance movement, transaction type, and suspicious account behavior. Investigators should review whether the origin account was rapidly depleted, whether the transaction amount is unusually high, and whether the transaction type matches common fraud patterns such as TRANSFER or CASH_OUT.
"""

    file_path = OUTPUT_DIR / f"fraud_case_{int(row['transaction_key'])}.txt"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(summary.strip())

print("Fraud case summaries generated successfully.")