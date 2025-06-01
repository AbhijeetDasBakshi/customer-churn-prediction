import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class TelcoDataPreprocessor:
    """
    A clean and simple preprocessor for IBM Telco Customer Churn dataset
    """
    
    def __init__(self, raw_data_path, processed_data_path=None):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path or "data/processed/telco_processed.csv"
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the raw telco dataset"""
        self.df = pd.read_csv(self.raw_data_path)
        print(f"Loaded data with shape: {self.df.shape}")
        return self
    
    def explore_data(self):
        """Quick data exploration"""
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        return self
    
    def clean_data(self):
        """Clean the dataset"""
        # Handle TotalCharges column (contains spaces instead of nulls)
        self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Drop rows with missing TotalCharges
        self.df = self.df.dropna(subset=['TotalCharges'])
        
        # Drop customerID as it's not useful for prediction
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
        
        return self
    
    def encode_features(self):
        """Simple binary encoding for categorical features only"""
        # Handle service-related columns with "No internet service" or "No phone service"
        service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                          'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        for col in service_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace('No internet service', 'No')
        
        # Handle MultipleLines
        if 'MultipleLines' in self.df.columns:
            self.df['MultipleLines'] = self.df['MultipleLines'].replace('No phone service', 'No')
        
        # Binary categorical variables (Yes/No) - Simple 1/0 encoding only
        binary_mappings = {
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
        
        # Apply binary mappings
        for col, mapping in binary_mappings.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].map(mapping)
        
        # Simple binary encoding for multi-category variables
        # InternetService: Fiber optic=1, others=0
        if 'InternetService' in self.df.columns:
            self.df['InternetService'] = (self.df['InternetService'] == 'Fiber optic').astype(int)
        
        # Contract: Month-to-month=1, others=0 (high churn risk)
        if 'Contract' in self.df.columns:
            self.df['Contract'] = (self.df['Contract'] == 'Month-to-month').astype(int)
        
        # PaymentMethod: Electronic check=1, others=0 (high churn risk)
        if 'PaymentMethod' in self.df.columns:
            self.df['PaymentMethod'] = (self.df['PaymentMethod'] == 'Electronic check').astype(int)
        
        # Encode target variable
        if 'Churn' in self.df.columns:
            self.df['Churn'] = self.df['Churn'].map({'Yes': 1, 'No': 0})
        
        return self
    
    def scale_features(self):
        """Scale numerical features"""
        # Identify columns to scale (exclude target variable)
        cols_to_scale = [col for col in self.df.columns if col != 'Churn']
        
        # Apply standard scaling
        self.df[cols_to_scale] = self.scaler.fit_transform(self.df[cols_to_scale])
        
        return self
    
    def save_processed_data(self):
        """Save the processed dataset"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        
        # Save to CSV
        self.df.to_csv(self.processed_data_path, index=False)
        print(f"Processed data saved to: {self.processed_data_path}")
        print(f"Final dataset shape: {self.df.shape}")
        
        return self
    
    def get_processed_data(self):
        """Return the processed dataframe"""
        return self.df
    
    def process_pipeline(self, save_data=True, explore=True):
        """
        Complete preprocessing pipeline
        
        Args:
            save_data (bool): Whether to save processed data to file
            explore (bool): Whether to perform data exploration
        
        Returns:
            pandas.DataFrame: Processed dataset
        """
        # Execute pipeline
        self.load_data()
        
        if explore:
            self.explore_data()
        
        self.clean_data()
        self.encode_features()
        # self.scale_features()
        
        if save_data:
            self.save_processed_data()
        
        print("Preprocessing completed!")
        return self.df

# Convenience function for direct usage
def preprocess_telco_data(raw_data_path, processed_data_path=None, save_data=True):
    """
    Simple function to preprocess telco data with default settings
    
    Args:
        raw_data_path (str): Path to raw CSV file
        processed_data_path (str): Path to save processed data (optional)
        save_data (bool): Whether to save processed data
    
    Returns:
        pandas.DataFrame: Processed dataset
    """
    preprocessor = TelcoDataPreprocessor(raw_data_path, processed_data_path)
    return preprocessor.process_pipeline(save_data=save_data)

# Example usage
if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    PROCESSED_DATA_PATH = "data/processed/telco_processed.csv"
    
    # Method 1: Using the class
    preprocessor = TelcoDataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    processed_df = preprocessor.process_pipeline()
    
    # Method 2: Using the convenience function
    # processed_df = preprocess_telco_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    print(f"\nPreprocessing complete!")
    print(f"Final dataset shape: {processed_df.shape}")
    print(f"Final columns: {list(processed_df.columns)}")