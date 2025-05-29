import pandas as pd

def load_raw_data():
    df = pd.read_csv("C:/Users/LENOVO/Documents/pymodel/customer-churn-prediction/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print("Data Shape:", df.shape)
    print(df.head())
    return df

if __name__ == "__main__":
    load_raw_data()
