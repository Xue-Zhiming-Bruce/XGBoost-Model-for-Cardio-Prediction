# gold_layer.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_for_analytics(silver_path):
    """Prepare silver data into the gold layer for analytics and ML"""
    # Create gold directory if it doesn't exist
    os.makedirs('datamart/gold', exist_ok=True)
    
    # Read from silver layer
    df = pd.read_parquet(silver_path)
    
    # Feature selection - keep only relevant features for modeling
    features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 
                'chol_1', 'chol_2', 'chol_3', 'gluc_1', 'gluc_2', 'gluc_3',
                'smoke', 'alco', 'active']
    target = 'cardio'
    
    # Create train/test datasets
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save train and test datasets to gold layer
    train_path = 'datamart/gold/cardio_train.parquet'
    test_path = 'datamart/gold/cardio_test.parquet'
    
    # Save with target included
    pd.concat([X_train, y_train], axis=1).to_parquet(train_path)
    pd.concat([X_test, y_test], axis=1).to_parquet(test_path)
    
    # Also save a feature-engineered version for direct model consumption
    feature_store_path = 'datamart/gold/cardio_feature_store.parquet'
    df[features + [target]].to_parquet(feature_store_path)
    
    print(f"Analytics-ready data saved to gold layer")
    return {'train': train_path, 'test': test_path, 'feature_store': feature_store_path}