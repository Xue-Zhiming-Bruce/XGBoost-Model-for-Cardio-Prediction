#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    brier_score_loss
)

# Import preprocessing modules
from utils.bronze_preprocess import ingest_raw_data
from utils.silver_preprocess import process_to_silver
from utils.gold_preprocess import prepare_for_analytics


def run_bronze_layer(input_file):
    """Run the bronze layer data processing"""
    print("\n=== Running Bronze Layer Processing ===")
    bronze_path = ingest_raw_data(input_file)
    return bronze_path


def run_silver_layer(bronze_path):
    """Run the silver layer data processing"""
    print("\n=== Running Silver Layer Processing ===")
    silver_path = process_to_silver(bronze_path)
    return silver_path


def run_gold_layer(silver_path):
    """Run the gold layer data processing"""
    print("\n=== Running Gold Layer Processing ===")
    gold_paths = prepare_for_analytics(silver_path)
    return gold_paths


def train_model(train_path, model_output_path=None):
    """Train an XGBoost model on the prepared data"""
    print("\n=== Training XGBoost Model ===")
    
    # Load training data
    train_data = pd.read_parquet(train_path)
    
    # Separate features and target
    features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 
                'chol_1', 'chol_2', 'chol_3', 'gluc_1', 'gluc_2', 'gluc_3',
                'smoke', 'alco', 'active']
    X_train = train_data[features]
    y_train = train_data['cardio']
    
    # Define XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train model
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=100)
    
    # Save model if path is provided
    if model_output_path:
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        model.save_model(model_output_path)
        print(f"Model saved to: {model_output_path}")
    
    return model


def evaluate_model(model, test_path):
    """Evaluate the trained model on test data"""
    print("\n=== Evaluating Model Performance ===")
    
    # Load test data
    test_data = pd.read_parquet(test_path)
    
    # Separate features and target
    features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 
                'chol_1', 'chol_2', 'chol_3', 'gluc_1', 'gluc_2', 'gluc_3',
                'smoke', 'alco', 'active']
    X_test = test_data[features]
    y_test = test_data['cardio']
    
    # Make predictions
    dtest = xgb.DMatrix(X_test)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('datamart/gold/confusion_matrix.png')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('datamart/gold/roc_curve.png')
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15)
    plt.title('Feature Importance')
    plt.savefig('datamart/gold/feature_importance.png')
    
    print("\nEvaluation plots saved to datamart/gold/ directory")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'brier': brier
    }


def make_prediction(model, input_data):
    """Make a prediction for a single patient"""
    # Convert input data to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    required_features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 
                         'chol_1', 'chol_2', 'chol_3', 'gluc_1', 'gluc_2', 'gluc_3',
                         'smoke', 'alco', 'active']
    
    for feature in required_features:
        if feature not in input_data.columns:
            raise ValueError(f"Missing required feature: {feature}")
    
    # Make prediction
    dmatrix = xgb.DMatrix(input_data[required_features])
    prediction_proba = model.predict(dmatrix)[0]
    prediction = 1 if prediction_proba > 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': prediction_proba,
        'risk_level': 'High' if prediction_proba > 0.7 else 'Medium' if prediction_proba > 0.3 else 'Low'
    }


def trigger_airflow_dag():
    """Trigger the Airflow DAG to run the pipeline"""
    print("\n=== Triggering Airflow DAG ===")
    try:
        # Check if Airflow is running
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", "name=airflow-webserver"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if not result.stdout.strip():
            print("Airflow is not running. Starting Airflow services...")
            subprocess.run(
                ["docker", "compose", "up", "-d", "airflow-webserver", "airflow-scheduler", "postgres"],
                check=True
            )
            print("Airflow services started. Please wait a moment for them to initialize.")
            print("You can access the Airflow UI at: http://localhost:8080")
        else:
            print("Airflow is already running.")
            
        # Trigger the DAG
        print("Triggering the cardio_detection_pipeline DAG...")
        subprocess.run(
            ["docker", "exec", "$(docker ps -q -f name=airflow-webserver)", "airflow", "dags", "trigger", "cardio_detection_pipeline"],
            shell=True,
            check=True
        )
        print("DAG triggered successfully. You can view the progress at: http://localhost:8080")
    except subprocess.CalledProcessError as e:
        print(f"Error triggering Airflow DAG: {e}")
        print("You can manually access Airflow at http://localhost:8080 if it's running.")


def run_full_pipeline(input_file, model_output_path=None):
    """Run the full data processing and modeling pipeline"""
    print("Starting full cardiovascular disease prediction pipeline...")
    
    # Run medallion architecture data processing
    bronze_path = run_bronze_layer(input_file)
    silver_path = run_silver_layer(bronze_path)
    gold_paths = run_gold_layer(silver_path)
    
    # Train and evaluate model
    model = train_model(gold_paths['train'], model_output_path)
    metrics = evaluate_model(model, gold_paths['test'])
    
    print("\n=== Pipeline Complete ===")
    print(f"Data processed through Bronze, Silver, and Gold layers")
    print(f"Model trained with accuracy: {metrics['accuracy']:.4f} and ROC AUC: {metrics['roc_auc']:.4f}")
    
    if model_output_path:
        print(f"Model saved to: {model_output_path}")
    
    # Trigger Airflow DAG
    trigger_airflow_dag()
    
    return model, metrics


def main():
    """Main function to parse arguments and run the appropriate pipeline stage"""
    parser = argparse.ArgumentParser(description='Cardiovascular Disease Prediction Pipeline')
    parser.add_argument('--input-file', type=str, default='cardio_train.csv',
                        help='Path to the input CSV file (default: cardio_train.csv)')
    parser.add_argument('--model-output', type=str, default='datamart/models/xgboost_model.json',
                        help='Path to save the trained model (default: datamart/models/xgboost_model.json)')
    parser.add_argument('--stage', type=str, choices=['bronze', 'silver', 'gold', 'train', 'evaluate', 'full'],
                        default='full', help='Pipeline stage to run (default: full)')
    parser.add_argument('--bronze-path', type=str, help='Path to bronze data (required for silver stage)')
    parser.add_argument('--silver-path', type=str, help='Path to silver data (required for gold stage)')
    parser.add_argument('--train-path', type=str, help='Path to training data (required for train stage)')
    parser.add_argument('--test-path', type=str, help='Path to test data (required for evaluate stage)')
    parser.add_argument('--model-path', type=str, help='Path to trained model (required for evaluate stage)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs('datamart/bronze', exist_ok=True)
    os.makedirs('datamart/silver', exist_ok=True)
    os.makedirs('datamart/gold', exist_ok=True)
    os.makedirs('datamart/models', exist_ok=True)
    
    # Run the appropriate pipeline stage
    if args.stage == 'bronze':
        run_bronze_layer(args.input_file)
    
    elif args.stage == 'silver':
        if not args.bronze_path:
            parser.error("--bronze-path is required for silver stage")
        run_silver_layer(args.bronze_path)
    
    elif args.stage == 'gold':
        if not args.silver_path:
            parser.error("--silver-path is required for gold stage")
        run_gold_layer(args.silver_path)
    
    elif args.stage == 'train':
        if not args.train_path:
            parser.error("--train-path is required for train stage")
        train_model(args.train_path, args.model_output)
    
    elif args.stage == 'evaluate':
        if not args.test_path or not args.model_path:
            parser.error("--test-path and --model-path are required for evaluate stage")
        model = xgb.Booster()
        model.load_model(args.model_path)
        evaluate_model(model, args.test_path)
    
    elif args.stage == 'full':
        run_full_pipeline(args.input_file, args.model_output)


if __name__ == "__main__":
    main()