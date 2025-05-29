import pandas as pd
from sqlalchemy import create_engine
import psycopg2

# Load CSV
df = pd.read_csv("C:/Users/LENOVO/Documents/pymodel/customer-churn-prediction/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Create PostgreSQL engine (adjust creds)
engine = create_engine("postgresql+psycopg2://postgres:1234@localhost:5432/customer_churn")

# Insert into DB
df.to_sql("churn_data", engine, index=False, if_exists="replace")

print("Data loaded into PostgreSQL successfully.")
