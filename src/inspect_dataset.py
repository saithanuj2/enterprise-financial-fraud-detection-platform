import pandas as pd

df = pd.read_csv("data/raw/paysim.csv")

print("=" * 60)
print("Dataset Shape")
print("=" * 60)
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFraud Distribution:")
print(df["isFraud"].value_counts())