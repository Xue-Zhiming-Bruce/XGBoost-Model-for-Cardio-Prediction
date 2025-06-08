# silver_layer.py
import pandas as pd
import os

def process_to_silver(bronze_path):
    """Process bronze data into the silver layer with cleaning and transformations"""
    # Create silver directory if it doesn't exist
    os.makedirs('datamart/silver', exist_ok=True)
    
    # Read from bronze layer
    df = pd.read_parquet(bronze_path)
    
    # Data cleaning and transformations
    # 1. Convert age from days to years
    df['age_years'] = df['age'] / 365
    
    # 2. Create BMI feature
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # 3. Handle outliers
    # Remove physiologically impossible values
    df = df[(df['ap_hi'] > 0) & (df['ap_lo'] > 0) & 
            (df['height'] > 0) & (df['weight'] > 0)]
    
    # 4. Encode categorical features
    # Binary encoding for gender (already done if 1/2)
    # One-hot encoding for cholesterol and gluc
    df = pd.get_dummies(df, columns=['cholesterol', 'gluc'], prefix=['chol', 'gluc'])
    
    # Save to silver layer
    silver_path = 'datamart/silver/cardio_cleaned.parquet'
    df.to_parquet(silver_path)
    
    print(f"Processed data saved to silver layer: {silver_path}")
    return silver_path