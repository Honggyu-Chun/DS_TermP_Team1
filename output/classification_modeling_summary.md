# Classification Modeling Summary

## 1. Purpose

The purpose of the classification modeling stage is to predict hotel booking cancellation risk using the target variable `is_canceled`.

Target definition:

- `0`: booking was not canceled
- `1`: booking was canceled

Business objective:

- Identify high cancellation-risk bookings before arrival.
- Support early customer follow-up, overbooking decisions, and resource planning.
- Provide an interpretable baseline model that connects EDA findings to business decisions.

Implementation files:

| File | Purpose |
| --- | --- |
| `notebooks/03_classification_modeling.ipynb` | Notebook-based modeling experiment and result review |
| `src/classification.py` | Reusable top-level Scikit-learn style modeling function |

Top-level function:

```python
run_classification_modeling(
    data_path="data/processed/hotel_bookings_clf.csv",
    output_dir="output/figures",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_figures=True,
    run_tuning=True,
    drop_assigned_room_type=True,
)
```

## 2. Reason

Classification is required because the project proposal defines cancellation prediction as one of the main analytical goals. From a business perspective, cancellation risk is not just an accuracy problem. Hotels need a balanced model that can find many actual cancellations while avoiding too many false alarms.

The stage also satisfies the course and project requirements:

- Includes classification.
- Uses encoded categorical variables from preprocessing.
- Applies scaling for scale-sensitive models through a Scikit-learn pipeline.
- Uses K-fold cross validation for classification.
- Uses only course-covered models: Logistic Regression, Decision Tree, and Random Forest.
- Avoids advanced external models such as XGBoost, LightGBM, CatBoost, and deep learning.

## 3. Logic

The modeling logic is:

1. Use the processed classification dataset `data/processed/hotel_bookings_clf.csv`.
2. Set `is_canceled` as the target.
3. Remove `is_canceled` from features to avoid target leakage.
4. Remove `adr` because it is the regression target and should not mix the classification and regression tasks.
5. Remove `assigned_room_type_*` by default because assigned room type may not be known at booking-time prediction.
6. Preserve the target ratio in train/test splitting with stratification.
7. Compare baseline models with Stratified K-fold cross validation.
8. Select the best baseline by F1-score.
9. Tune the best baseline model with `GridSearchCV(scoring="f1")`.
10. Evaluate the final model on the untouched test set.

This logic avoids columns that may not be available at prediction time and keeps the classification objective aligned with the proposal.

## 4. Method

Input dataset status:

| Item | Value |
| --- | ---: |
| Rows | 119,210 |
| Total columns before modeling | 83 |
| Input features after exclusions | 71 |
| Missing values | 0 |
| Object/string columns | 0 |
| Not canceled `0` | 75,011, 62.92% |
| Canceled `1` | 44,199, 37.08% |

Excluded columns:

| Excluded column group | Reason |
| --- | --- |
| `is_canceled` | Target variable |
| `adr` | Regression target, excluded to keep tasks separated |
| `assigned_room_type_*` | Removed by default because it may not be available at prediction time |

Compared models:

| Model | Method detail |
| --- | --- |
| Logistic Regression | Uses `StandardScaler` inside a `Pipeline` |
| Decision Tree | Tree-based nonlinear baseline |
| Random Forest | Ensemble tree baseline |

Evaluation setup:

- Train/test split: 80:20.
- Test split stratification: `stratify=y`.
- Cross validation: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Metrics: accuracy, precision, recall, F1-score, and confusion matrix.
- Main selection metric: F1-score.
- Hyperparameter tuning: `GridSearchCV` on the best baseline model.

The Logistic Regression scaler is fit only inside training folds during cross validation and only on the training set during final evaluation. Decision Tree and Random Forest do not require scaling.

## 5. Results

Baseline 5-fold cross validation:

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8401 | 0.8142 | 0.7371 | 0.7737 |
| Random Forest | 0.8477 | 0.8645 | 0.6989 | 0.7729 |
| Logistic Regression | 0.8075 | 0.8029 | 0.6373 | 0.7105 |

Decision Tree and Random Forest were very close, but Decision Tree had the best baseline F1-score and stronger recall. It was selected as the tuning candidate.

GridSearchCV search space:

```python
{
    "max_depth": [8, 10, 12, 15],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}
```

Best parameters:

```text
criterion='gini', max_depth=15, min_samples_split=5
```

Final selected model: **Decision Tree (Tuned)**

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8511 |
| Precision | 0.8161 |
| Recall | 0.7723 |
| F1-score | 0.7936 |

Confusion matrix:

| Actual / Predicted | Predicted not canceled | Predicted canceled |
| --- | ---: | ---: |
| Actual not canceled | 13,464 | 1,538 |
| Actual canceled | 2,013 | 6,827 |

Top features:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `deposit_type_Non Refund` | 0.3658 |
| 2 | `market_segment_Online TA` | 0.0973 |
| 3 | `lead_time` | 0.0870 |
| 4 | `total_of_special_requests` | 0.0803 |
| 5 | `country_PRT` | 0.0651 |

Output figures:

| Figure | Path |
| --- | --- |
| Model comparison | `output/figures/classification_model_comparison.png` |
| Confusion matrix | `output/figures/confusion_matrix_decision_tree.png` |
| Feature importance | `output/figures/classification_feature_importance.png` |

## 6. Function Output Specification

`run_classification_modeling()` returns:

| Key | Description |
| --- | --- |
| `cv_result` | Cross validation metrics for all baseline models |
| `test_result` | Final holdout test metrics for the selected model |
| `confusion_matrix` | Final confusion matrix as a labeled DataFrame |
| `feature_importance` | Feature importance or absolute coefficient values |
| `best_model_name` | Name of the selected model |
| `best_model` | Fitted Scikit-learn estimator or pipeline |
| `feature_columns` | Final input feature names |
| `dropped_columns` | Columns excluded before modeling |

Usage:

```python
from src.classification import run_classification_modeling

result = run_classification_modeling()
print(result["best_model_name"])
print(result["cv_result"].round(4))
print(result["test_result"].round(4))
```

## 7. Stage Conclusion

The classification modeling stage is consistent with the proposal and project requirements. It uses a leakage-aware feature set, encoded categorical variables, model-specific scaling, Stratified K-fold cross validation, and F1-based final model selection. The final tuned Decision Tree reaches test F1-score 0.7936 and provides interpretable cancellation-risk drivers.
