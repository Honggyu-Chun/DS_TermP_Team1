# DS_TermP_Team1: Hotel Booking Demand Prediction

## Project Overview

This repository contains Team 1's Data Science term project using the Kaggle Hotel Booking Demand dataset. The project predicts hotel booking cancellation risk and supports ADR-based revenue management.

Business objective:

- Classification: predict cancellation risk using `is_canceled`.
- Regression: predict average daily rate using `adr`.
- Practical use: support cancellation management, overbooking decisions, and room revenue planning.

Dataset:

- Kaggle Hotel Booking Demand
- Link: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

## Folder Architecture

```text
DS_TermP_Team1/
|-- data/
|   |-- raw/
|   |   `-- hotel_bookings.csv
|   `-- processed/
|       |-- hotel_bookings_clf.csv
|       `-- hotel_bookings_reg.csv
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_processing.ipynb
|   |-- 03_classification_modeling.ipynb
|   `-- 04_regression_modeling.ipynb
|-- reports/
|   |-- figures/
|   |-- eda_summary.md
|   |-- preprocessing_summary.md
|   |-- classification_modeling_summary.md
|   `-- regression_modeling_summary.md
|-- src/
|   |-- __init__.py
|   |-- processing.py
|   |-- classification.py
|   |-- regression.py
|   `-- pipeline.py
|-- requirements.txt
`-- README.md
```

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run Preprocessing

```bash
python src/processing.py
```

Or use the top-level function:

```python
from src.processing import preprocess_hotel_booking_data

result = preprocess_hotel_booking_data()
print(result["classification_path"])
print(result["regression_path"])
```

Preprocessing creates:

- `data/processed/hotel_bookings_clf.csv`
- `data/processed/hotel_bookings_reg.csv`

The preprocessing step handles missing values, removes zero-guest bookings, creates derived features, removes leakage-prone reservation status columns, and applies one-hot encoding.

## How to Run Classification Modeling

```bash
python src/classification.py
```

Or use the top-level function:

```python
from src.classification import run_classification_modeling

result = run_classification_modeling()
print(result["best_model_name"])
print(result["cv_result"].round(4))
print(result["test_result"].round(4))
```

Classification details:

- Target: `is_canceled`
- Excluded columns: `is_canceled`, `adr`, and `assigned_room_type_*` by default
- Models: Logistic Regression, Decision Tree, Random Forest
- Scaling: `StandardScaler` inside the Logistic Regression pipeline
- Cross validation: Stratified K-fold CV
- Final selection criterion: F1-score with GridSearchCV tuning for the best baseline model

## How to Run Regression Modeling

```bash
python src/regression.py
```

Or use the top-level function:

```python
from src.regression import run_regression_modeling

result = run_regression_modeling()
print(result["best_model_name"])
print(result["test_result"].round(4))
```

Regression details:

- Target: `adr`
- Dataset: non-canceled bookings with ADR filtering
- Models: Linear Regression, Decision Tree variants, Random Forest
- Scaling: `StandardScaler` inside the Linear Regression pipeline
- Cross validation: K-fold CV
- Final selection criterion: lowest CV RMSE

## Full Pipeline

The full pipeline is optional and is not required for notebook execution.

```python
from src.pipeline import run_full_pipeline

result = run_full_pipeline()
```

Command line:

```bash
python src/pipeline.py
```

## Top-level Function List

| Function | Location | Purpose |
| --- | --- | --- |
| `preprocess_hotel_booking_data()` | `src/processing.py` | Create processed classification and regression datasets |
| `run_classification_modeling()` | `src/classification.py` | Train and evaluate cancellation classification models |
| `run_regression_modeling()` | `src/regression.py` | Train and evaluate ADR regression models |
| `run_full_pipeline()` | `src/pipeline.py` | Run preprocessing, classification, and regression from one entry point |

## Output Files and Figures

Stage summaries:

- `reports/eda_summary.md`
- `reports/preprocessing_summary.md`
- `reports/classification_modeling_summary.md`
- `reports/regression_modeling_summary.md`

Processed data:

- `data/processed/hotel_bookings_clf.csv`
- `data/processed/hotel_bookings_reg.csv`

Classification figures:

- `reports/figures/classification_model_comparison.png`
- `reports/figures/confusion_matrix_decision_tree.png`
- `reports/figures/classification_feature_importance.png`

Regression figures:

- `reports/figures/regression_model_comparison.png`
- `reports/figures/regression_actual_vs_predicted.png`
- `reports/figures/regression_residual_analysis.png`
- `reports/figures/regression_feature_importance.png`

## Project Requirements Coverage

- Data scaling: implemented through Scikit-learn pipelines for scale-sensitive linear models.
- Encoding: categorical variables are one-hot encoded in preprocessing.
- Classification: cancellation prediction using `is_canceled`.
- Regression: ADR prediction using `adr`.
- K-fold CV for classification: implemented with `StratifiedKFold`.
- Open Source SW readiness: reusable top-level functions, package import support, requirements file, manuals, and clear run instructions.
- Model scope: limited to course-covered Scikit-learn models. XGBoost, LightGBM, CatBoost, and deep learning models are not used.

## GitHub URL

Placeholder:

```text
https://github.com/<organization-or-user>/DS_TermP_Team1
```

## Citation and References

- Antonio, N., Almeida, A., and Nunes, L. (2019). Hotel booking demand datasets.
- Kaggle Hotel Booking Demand dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
