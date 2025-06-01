import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler

# ----- CONFIG -----
PG_HOST = "localhost"
PG_PORT = "5432"
PG_DB = "customer_churn"
PG_USER = "postgres"
PG_PASS = "1234"
RAW_TABLE = "churn_data"                  # existing raw table
PROCESSED_TABLE = "churn_data_processed"  # new table to store processed output

# ----- CONNECT -----
conn_string = f"postgresql://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}"
engine = create_engine(conn_string)
conn = engine.connect()

# ----- LOAD RAW DATA -----
df = pd.read_sql(f"SELECT * FROM {RAW_TABLE}", conn)
print(f"[INFO] Loaded data from PostgreSQL table: {df.shape}")

# ----- CLEAN DATA -----
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# ----- ENCODE FEATURES -----
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    if col in df.columns:
        df[col] = df[col].replace('No internet service', 'No')

if 'MultipleLines' in df.columns:
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

binary_map = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'MultipleLines': {'Yes': 1, 'No': 0},
    'OnlineSecurity': {'Yes': 1, 'No': 0},
    'OnlineBackup': {'Yes': 1, 'No': 0},
    'DeviceProtection': {'Yes': 1, 'No': 0},
    'TechSupport': {'Yes': 1, 'No': 0},
    'StreamingTV': {'Yes': 1, 'No': 0},
    'StreamingMovies': {'Yes': 1, 'No': 0}
}
for col, mapping in binary_map.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

if 'InternetService' in df.columns:
    df['InternetService'] = (df['InternetService'] == 'Fiber optic').astype(int)
if 'Contract' in df.columns:
    df['Contract'] = (df['Contract'] == 'Month-to-month').astype(int)
if 'PaymentMethod' in df.columns:
    df['PaymentMethod'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
if 'Churn' in df.columns:
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# OPTIONAL: scale features if needed
# scaler = StandardScaler()
# feature_cols = [col for col in df.columns if col != 'Churn']
# df[feature_cols] = scaler.fit_transform(df[feature_cols])

# ----- SAVE TO NEW TABLE -----
with engine.begin() as connection:
    connection.execute(text(f"DROP TABLE IF EXISTS {PROCESSED_TABLE}"))  # <-- Wrap with `text`
    df.to_sql(PROCESSED_TABLE, connection, index=False, if_exists='replace')

print(f"[Done] Processed data saved to PostgreSQL table: {PROCESSED_TABLE}")
print(f"[INFO] Final processed shape: {df.shape}")
print(df.head())
