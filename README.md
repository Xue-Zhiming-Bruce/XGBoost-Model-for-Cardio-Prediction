# XGBoost Model For Cardio Prediction

This project focuses on building and evaluating an XGBoost machine learning model to predict the presence of cardiovascular disease based on patient examination data.

## Project Overview

The primary goal is to leverage a dataset containing patient attributes and health metrics to train a classifier capable of identifying individuals at risk of cardiovascular disease. The project involves several stages, including data cleaning, exploratory data analysis (EDA), feature engineering, model training with hyperparameter considerations (like class weighting), and comprehensive model evaluation.

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Running the Project

#### Option 1: Using Docker Compose

1. Start the services (Jupyter and Airflow):

```bash
docker compose up
```

2. Access Jupyter Lab:
   - Open your browser and navigate to: http://localhost:8888
   - No password is required
   - You can use Jupyter to explore the data, run notebooks, and develop models

3. Access Airflow Web UI:
   - Open your browser and navigate to: http://localhost:8080
   - Default credentials: username: `airflow`, password: `airflow`
   - You can monitor and trigger DAG runs from the Airflow UI

#### Option 2: Running the Pipeline Directly

You can also run the pipeline directly using the main.py script:

```bash
python main.py --stage full
```

This will:
1. Process the data through all layers (Bronze, Silver, Gold)
2. Train an XGBoost model
3. Evaluate the model and save metrics and plots

After running the pipeline, you can access Airflow to view the DAG runs:

```bash
docker compose up airflow-webserver airflow-scheduler postgres
```

Then navigate to http://localhost:8080 in your browser.

## Dataset

The model was trained using the `cardio_train.csv` dataset, which is part of the Cardiovascular Disease dataset available on Kaggle.

* **Original Dataset Link:** [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Methodology

**Class Weight Tuning**: 

For this cardiovascular disease prediction model, we implemented class weight tuning in XGBoost using the `scale_pos_weight` parameter. This approach was chosen specifically for medical predictions, where higher recall (correctly identifying true positive cases) is often crucial, even if it slightly impacts precision.

We set `scale_pos_weight` to 1.5 (ratio of 3:2) for the positive class, which provides an optimal balance between improving recall and maintaining overall model performance. This weighting gives more importance to correctly classifying patients with cardiovascular disease, reducing the chance of missing potential cases that require medical attention.

![Performance Metrics for Different Weight Ratios](Weight%20Ratio%20Performances.png)

The chart above shows how different weight ratios affect model performance metrics. Note how the 1:2 ratio significantly increases recall but at the cost of precision and overall accuracy. The 3:2 ratio (1.5) we selected offers a balanced improvement in recall while maintaining good overall performance.

## Key Libraries Used

* pandas
* numpy
* matplotlib
* seaborn
* xgboost
* scikit-learn
* statsmodels

## Considerations for Medical Prediction

* **Recall Emphasis**: In diagnosing conditions like cardiovascular disease, correctly identifying patients who *have* the disease (True Positives) is often prioritized over minimizing False Positives. A higher recall means fewer missed cases, which is critical for timely intervention.
* **Weight Tuning**: The process of adjusting `scale_pos_weight` directly addresses this need by giving more importance to correctly classifying the positive (cardiovascular disease) class, thereby boosting recall.

## Results Summary

The cross-validation results using the tuned model (scale_pos_weight=3/2) indicated the following mean performance:
* Accuracy: ~0.7259
* Precision: ~0.7015
* Recall: ~0.7769
* F1-Score: ~0.7373

On the test set, the model achieved an ROC AUC of ~0.80 and a Brier Score of ~0.1842. The feature importance analysis highlighted systolic blood pressure (`ap_hi`), cholesterol levels (`cholesterol_3`), and age as the top predictors. The weight tuning successfully increased recall compared to an unweighted model, aligning with the goal of minimizing missed diagnoses in a medical context.

## Model Metrics

After running the pipeline, model evaluation metrics and plots are saved to the `datamart/gold/` directory:

- Confusion matrix: `confusion_matrix.png`
- ROC curve: `roc_curve.png`
- Feature importance: `feature_importance.png`

Metrics are also printed to the console, including:
- Accuracy
- F1 Score
- Precision
- Recall
- ROC AUC
- Brier Score

## Airflow DAG

The Airflow DAG (`cardio_detection_pipeline`) includes the following tasks:

1. `start_pipeline`: Dummy task to mark the start
2. `bronze_process_raw_data`: Ingest and process raw data
3. `silver_clean_data`: Clean and transform the data
4. `gold_prepare_model_data`: Prepare data for analytics and modeling
5. `end_pipeline`: Dummy task to mark the end
