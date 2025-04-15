# XGBoost Model For Cardio Prediction

This project focuses on building and evaluating an XGBoost machine learning model to predict the presence of cardiovascular disease based on patient examination data.

## Project Overview

The primary goal is to leverage a dataset containing patient attributes and health metrics to train a classifier capable of identifying individuals at risk of cardiovascular disease. The project involves several stages, including data cleaning, exploratory data analysis (EDA), feature engineering, model training with hyperparameter considerations (like class weighting), and comprehensive model evaluation.

## Dataset

The model was trained using the `cardio_train.csv` dataset.

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

## How to Use

(Assuming the code is available in a repository)
1.  Clone the repository.
2.  Install the required libraries (e.g., using `pip install -r requirements.txt`).
3.  Ensure the `cardio_train.csv` dataset is accessible.
4.  Run the Jupyter Notebook or Python script to reproduce the analysis and model training.
