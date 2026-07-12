import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta

OUTPUT_PATH = Path("data/raw/fraud_transactions.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)

NUM_RECORDS = 50000

merchant_categories = [
    "Grocery", "Electronics", "Travel", "Gas", "Restaurant",
    "Online Retail", "Luxury", "Healthcare", "Entertainment"
]

transaction_types = ["POS", "Online", "ATM", "Transfer"]

locations = [
    "New York", "Chicago", "Los Angeles", "Dallas", "Miami",
    "San Francisco", "Seattle", "Boston", "Atlanta", "Denver"
]

def generate_data(num_records: int) -> pd.DataFrame:
    rows = []

    start_date = datetime(2025, 1, 1)

    for i in range(1, num_records + 1):
        customer_id = random.randint(1000, 9999)
        account_id = random.randint(100000, 999999)
        merchant_id = random.randint(5000, 9999)
        device_id = random.randint(1000000, 9999999)

        transaction_timestamp = start_date + timedelta(
            days=random.randint(0, 364),
            minutes=random.randint(0, 1440)
        )

        transaction_amount = round(np.random.exponential(scale=85), 2)

        merchant_category = random.choice(merchant_categories)
        transaction_type = random.choice(transaction_types)

        customer_home_location = random.choice(locations)
        transaction_location = random.choice(locations)

        is_international = np.random.choice([0, 1], p=[0.92, 0.08])
        is_new_device = np.random.choice([0, 1], p=[0.85, 0.15])
        failed_login_count = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.55, 0.22, 0.11, 0.06, 0.04, 0.02])
        account_age_days = random.randint(10, 2500)

        amount_deviation_score = round(np.random.normal(loc=1.0, scale=0.45), 2)

        fraud_probability = 0.02

        if transaction_amount > 300:
            fraud_probability += 0.10
        if merchant_category in ["Luxury", "Electronics", "Travel"]:
            fraud_probability += 0.06
        if is_international == 1:
            fraud_probability += 0.12
        if is_new_device == 1:
            fraud_probability += 0.10
        if failed_login_count >= 3:
            fraud_probability += 0.15
        if transaction_location != customer_home_location:
            fraud_probability += 0.08
        if account_age_days < 90:
            fraud_probability += 0.06
        if amount_deviation_score > 1.6:
            fraud_probability += 0.08

        fraud_label = np.random.choice([0, 1], p=[1 - min(fraud_probability, 0.85), min(fraud_probability, 0.85)])

        rows.append({
            "transaction_id": i,
            "customer_id": customer_id,
            "account_id": account_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "transaction_timestamp": transaction_timestamp,
            "transaction_amount": transaction_amount,
            "transaction_type": transaction_type,
            "merchant_category": merchant_category,
            "transaction_location": transaction_location,
            "customer_home_location": customer_home_location,
            "is_international": is_international,
            "is_new_device": is_new_device,
            "failed_login_count": failed_login_count,
            "account_age_days": account_age_days,
            "amount_deviation_score": amount_deviation_score,
            "fraud_label": fraud_label
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_data(NUM_RECORDS)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset created successfully: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Fraud Rate: {round(df['fraud_label'].mean() * 100, 2)}%")
    print(df.head())