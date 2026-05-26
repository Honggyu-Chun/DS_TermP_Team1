# Classification Modeling User Manual

## 1. Purpose

This document explains how to use the top-level classification modeling function for the Hotel Booking Demand term project. The function wraps data loading, feature selection, model comparison, hyperparameter tuning, final test evaluation, and figure generation in a reusable Scikit-learn style interface.

The classification objective is to predict hotel booking cancellation risk using the target variable `is_canceled`.

## 2. Function Location

```text
src/classification.py
```

Top-level function:

```python
run_classification_modeling(
    data_path="data/processed/hotel_bookings_clf.csv",
    output_dir="reports/figures",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_figures=True,
    run_tuning=True,
    drop_assigned_room_type=True,
)
```

## 3. Input Dataset

Default input file:

```text
data/processed/hotel_bookings_clf.csv
```

This dataset is created by `src/processing.py`.

| Item | Description |
| --- | --- |
| Dataset | Preprocessed classification dataset |
| Target | `is_canceled` |
| Missing values | Handled during preprocessing |
| Categorical encoding | One-hot encoding during preprocessing |
| Scaling | Applied inside the Logistic Regression pipeline |
| Main business use | Cancellation risk prediction |

## 4. Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `data_path` | `data/processed/hotel_bookings_clf.csv` | CSV file used for classification modeling |
| `output_dir` | `reports/figures` | Directory where output figures are saved |
| `test_size` | `0.2` | Test set ratio for the final holdout evaluation |
| `n_splits` | `5` | Number of folds for Stratified K-fold cross validation |
| `random_state` | `42` | Random seed for reproducible splits and model training |
| `save_figures` | `True` | Whether to save model comparison, confusion matrix, and feature importance figures |
| `run_tuning` | `True` | Whether to run GridSearchCV for the best baseline model |
| `drop_assigned_room_type` | `True` | Whether to remove `assigned_room_type_*` columns to reduce prediction-time leakage risk |

## 5. Returned Dictionary

The function returns a dictionary with the following keys.

| Key | Description |
| --- | --- |
| `cv_result` | Cross validation results for Logistic Regression, Decision Tree, and Random Forest |
| `test_result` | Final holdout test metrics for the selected model |
| `confusion_matrix` | Final confusion matrix as a labeled DataFrame |
| `feature_importance` | Feature importance or absolute coefficient values for the selected model |
| `best_model_name` | Name of the selected model |
| `best_model` | Fitted Scikit-learn estimator or pipeline |
| `feature_columns` | Input feature column names used by the model |
| `dropped_columns` | Columns removed before modeling |

## 6. Generated Output Figures

When `save_figures=True`, the function saves:

| File | Description |
| --- | --- |
| `classification_model_comparison.png` | Cross validation comparison across accuracy, precision, recall, and F1 |
| `confusion_matrix_decision_tree.png` | Confusion matrix for the final model when Decision Tree is selected |
| `classification_feature_importance.png` | Top 15 feature importance or coefficient values |

If a different final model is selected, the confusion matrix file name uses that model name in lowercase snake case.

## 7. Modeling Specification

The target variable is:

```text
is_canceled
```

The following columns are excluded from the model input:

| Excluded column group | Reason |
| --- | --- |
| `is_canceled` | Target variable |
| `adr` | Regression target, excluded to keep classification and regression tasks separate |
| `assigned_room_type_*` | Removed by default because assigned room type may be unavailable at booking-time prediction |

The model candidates are limited to course-covered Scikit-learn models:

| Model | Notes |
| --- | --- |
| Logistic Regression | Uses `StandardScaler` inside a `Pipeline` |
| Decision Tree | Tree-based baseline |
| Random Forest | Ensemble baseline with internal parallelism disabled during outer CV to avoid oversubscription |

## 8. Scaling and Encoding Strategy

Categorical encoding is completed during preprocessing using one-hot encoding.

Scaling is not applied to the full dataset before splitting. Logistic Regression uses a Scikit-learn `Pipeline` with `StandardScaler`, so scaling is fit only on each training fold during cross validation and only on the training set during final evaluation. Decision Tree and Random Forest do not require scaling.

## 9. K-fold Cross Validation Strategy

The function uses:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

The number of folds and random seed can be changed through `n_splits` and `random_state`. Stratification preserves the cancellation class ratio across folds.

## 10. Final Model Selection Criterion

The baseline models are compared by mean cross validation F1-score. The best baseline model is then tuned with `GridSearchCV(scoring="f1")` when `run_tuning=True`.

The final model is selected by F1-score because cancellation prediction needs a balance between precision and recall. With the current project setup, the tuned Decision Tree remains the expected final classification model.

## 11. Usage Example

Run from the project root:

```python
from src.classification import run_classification_modeling

result = run_classification_modeling()

print(result["best_model_name"])
print(result["cv_result"].round(4))
print(result["test_result"].round(4))
```

Command line execution:

```bash
python src/classification.py
```
