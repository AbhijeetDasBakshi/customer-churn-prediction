import pandas as pd
import numpy as np
# from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient

# ----- CONFIG -----
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "customer_churn"
RAW_COLLECTION = "data"            # your existing MongoDB collection
PROCESSED_COLLECTION = "data_processed"  # new collection to store processed data

# ----- CONNECT TO MONGO -----
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
raw_collection = db[RAW_COLLECTION]
processed_collection = db[PROCESSED_COLLECTION]

# ----- LOAD DATA FROM MONGO -----
df = pd.DataFrame(list(raw_collection.find()))
if df.empty:
    raise ValueError(" No data found in MongoDB collection!")
if '_id' in df.columns:
    df.drop('_id', axis=1, inplace=True)
print(f"[INFO] Loaded data from MongoDB: {df.shape}")

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

# ----- SAVE TO PROCESSED COLLECTION -----
processed_collection.delete_many({})
processed_collection.insert_many(df.to_dict(orient='records'))
print(f"[DONE] Processed data saved to MongoDB collection: {PROCESSED_COLLECTION}")
print(f"[INFO] Final processed shape: {df.shape}")
print(df.head())
