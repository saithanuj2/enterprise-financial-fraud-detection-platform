import pandas as pd
import joblib
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

DB_USER = "fraud_user"
DB_PASSWORD = "fraud_pass"
DB_HOST = "127.0.0.1"
DB_PORT = "55432"
DB_NAME = "fraud_warehouse"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Reading fraud_features from PostgreSQL...")

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
"""

df = pd.read_sql(query, engine)

print(f"Rows loaded: {len(df)}")

df_encoded = pd.get_dummies(df, columns=["transaction_type"], drop_first=True)

X = df_encoded.drop(columns=["transaction_key", "is_fraud"])
y = df_encoded["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Generating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nROC AUC:")
print(roc_auc_score(y_test, y_prob))

joblib.dump(model, MODEL_DIR / "fraud_detection_model.pkl")

print("\nModel saved to models/fraud_detection_model.pkl")

predictions = X_test.copy()
predictions["fraud_probability"] = y_prob
predictions["predicted_fraud"] = y_pred
predictions["actual_fraud"] = y_test.values

predictions.to_sql(
    "fraud_model_predictions",
    engine,
    if_exists="replace",
    index=False
)

print("Predictions written to PostgreSQL table: fraud_model_predictions")
print("ML pipeline completed successfully.")