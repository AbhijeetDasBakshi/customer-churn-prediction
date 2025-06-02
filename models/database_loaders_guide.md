# Database Loaders for Customer Churn Prediction

Simple MongoDB and PostgreSQL loaders to replace CSV file loading.

## Installation

```bash
pip install pymongo psycopg2-binary sqlalchemy pandas scikit-learn
```

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from sqlalchemy import create_engine

# MongoDB Loader
def load_data_from_mongodb():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["customer_churn"]
    collection = db["data_processed"]
    
    data = list(collection.find({}))
    df = pd.DataFrame(data)
    
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# PostgreSQL Loader
def load_data_from_postgresql():
    connection_string = "postgresql://username:password@localhost:5432/customer_churn"
    engine = create_engine(connection_string)
    df = pd.read_sql("SELECT * FROM churn_data_processed", engine)
    
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# CSV Loader (original)
def load_data_from_csv():
    df = pd.read_csv("data/processed/telco_processed.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    
    return train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
```

## Usage

Replace your current loading code with:

```python
# Choose one:
X_train, X_test, y_train, y_test = load_data_from_mongodb()
X_train, X_test, y_train, y_test = load_data_from_postgresql()
X_train, X_test, y_train, y_test = load_data_from_csv()
```

## Configuration

**MongoDB:**
- Update connection string with your credentials
- Database: `customer_churn`
- Collection: `data_processed`

**PostgreSQL:**
- Update connection string with your credentials
- Table: `churn_data_processed`
