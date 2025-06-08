# bronze_layer.py
import pandas as pd
import os

def ingest_raw_data(file_path):
    """Ingest raw data into the bronze layer"""
    # Create bronze directory if it doesn't exist
    os.makedirs('datamart/bronze', exist_ok=True)
    
    # Read the raw CSV file with semicolon separator
    df = pd.read_csv(file_path, sep=';')
    
    # Save as-is to bronze layer (parquet for better performance)
    bronze_path = 'datamart/bronze/cardio_raw.parquet'
    df.to_parquet(bronze_path)
    
    print(f"Raw data saved to bronze layer: {bronze_path}")
    return bronze_path