import pandas as pd
from pymongo import MongoClient

# Load CSV
df = pd.read_csv("C:/Users/LENOVO/Documents/pymodel/customer-churn-prediction/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Convert DataFrame to dictionary records
records = df.to_dict(orient="records")

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["customer-churn"]
collection = db["data"]

# Insert records
collection.insert_many(records)

print("Data loaded into MongoDB successfully.")
