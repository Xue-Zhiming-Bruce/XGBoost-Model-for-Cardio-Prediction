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

1.  **Data Loading & Initial Setup**:
    * Loaded the dataset using pandas.
    * Imported necessary libraries including pandas, numpy, matplotlib, seaborn, xgboost, scikit-learn, and statsmodels.

2.  **Data Cleaning & Preprocessing**:
    * Removed 'id' column and duplicate rows. Checked for missing values (none found).
    * Transformed features: Converted 'age' to years, performed binary encoding for 'gender', and one-hot encoded 'cholesterol' and 'gluc'.
    * Engineered 'bmi' feature from 'height' and 'weight', then dropped original columns.
    * Filtered outliers based on physiological constraints for blood pressure ('ap_hi', 'ap_lo'), height, weight, and BMI.
    * Verified low multicollinearity among features using VIF.

3.  **Exploratory Data Analysis (EDA)**:
    * Generated histograms for all features to understand their distributions.
    * Created a correlation heatmap to visualize relationships between features.

4.  **Model Building & Evaluation**:
    * Split the data into training and testing sets (80/20 split) stratified by the target variable 'cardio'.
    * **Class Weight Tuning**: Explicitly tested different `scale_pos_weight` values (1:2, 2:3, 3:4, 4:5) in XGBoost. This tuning was performed recognizing that for medical predictions, higher recall (correctly identifying true positive cases) is often crucial, even if it slightly impacts precision. The ratio 3:2 (weight of 1.5 for the positive class) was selected for the final model as a balance between improving recall and maintaining overall performance.
    * **Cross-Validation**: Employed Stratified K-Fold cross-validation (5 folds) to get robust estimates of model performance.
    * **Model Training**: Trained an XGBoost Classifier with selected hyperparameters (e.g., learning_rate=0.1, max_depth=5, scale_pos_weight=3/2).
    * **Evaluation Metrics**: Evaluated the model using:
        * Accuracy, Precision, Recall, F1-Score (via cross-validation and on the test set).
        * ROC Curve and AUC score.
        * Brier Score for probability calibration.

5.  **Feature Importance**:
    * Calculated and visualized the importance of each feature in the trained XGBoost model. `ap_hi` (systolic blood pressure) was found to be the most influential feature.

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
