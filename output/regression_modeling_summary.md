# Regression Modeling Summary

## 1. Purpose

The purpose of the regression modeling stage is to predict `adr` (Average Daily Rate) for hotel bookings.

Business objective:

- Estimate expected room rate from booking conditions.
- Support hotel pricing and revenue management decisions.
- Understand how seasonality, hotel type, guest count, market segment, and booking patterns relate to ADR.

Implementation files:

| File | Purpose |
| --- | --- |
| `notebooks/04_regression_modeling.ipynb` | Notebook-based modeling experiment and result review |
| `src/regression.py` | Reusable top-level Scikit-learn style modeling function |

Top-level function:

```python
run_regression_modeling(
    data_path="data/processed/hotel_bookings_reg.csv",
    output_dir="output/figures",
    test_size=0.2,
    n_splits=5,
    random_state=42,
    save_figures=True,
)
```

## 2. Reason

Regression is required because the project proposal includes ADR prediction as the second modeling goal. While classification predicts whether a booking will be canceled, regression estimates expected booking value for completed/non-canceled reservations.

The stage also satisfies project requirements:

- Includes regression as the second modeling task.
- Uses encoded categorical variables from preprocessing.
- Applies scaling for scale-sensitive linear models through a Scikit-learn pipeline.
- Uses only course-covered models: Linear Regression, Decision Tree, and Random Forest.
- Avoids advanced external models such as XGBoost, LightGBM, CatBoost, and deep learning.

## 3. Logic

The modeling logic is:

1. Use the processed regression dataset `data/processed/hotel_bookings_reg.csv`.
2. Use only non-canceled bookings because ADR prediction is intended for completed reservations.
3. Apply the preprocessing ADR filter `0 < adr < 500` to reduce target outlier noise.
4. Set `adr` as the target.
5. Remove `adr` from features to avoid target leakage.
6. Remove `is_canceled` because it is constant 0 in the regression dataset.
7. Remove `assigned_room_type_*` because assigned room type may not be known before stay/check-in.
8. Compare a small set of course-covered regression models using K-fold cross validation.
9. Select the final model by the lowest CV RMSE.
10. Evaluate the selected model on the untouched test set.

This logic keeps the regression task separate from classification and supports prediction-time revenue management.

## 4. Method

Input dataset status:

| Item | Value |
| --- | ---: |
| Rows | 73,386 |
| Total columns before modeling | 83 |
| Input features after exclusions | 71 |
| Missing values | 0 |
| Object/string columns | 0 |
| Target | `adr` |
| `is_canceled` value | Constant 0 |
| ADR mean | 102.373 |
| ADR median | 94.500 |
| ADR standard deviation | 47.222 |
| ADR minimum | 0.260 |
| ADR maximum | 451.500 |

Excluded columns:

| Excluded column group | Reason |
| --- | --- |
| `adr` | Regression target |
| `is_canceled` | Constant zero in the regression dataset |
| `assigned_room_type_*` | Removed by default to reduce prediction-time leakage risk |

Compared model/parameter combinations:

| Model combination | Method detail |
| --- | --- |
| Linear Regression | Uses `StandardScaler` inside a `Pipeline` |
| Decision Tree depth 8 | Shallow nonlinear tree model |
| Decision Tree depth 12 | Medium-depth tree model |
| Decision Tree depth 16 | More flexible tree model |
| Random Forest depth 14 | Ensemble tree regression model |

Evaluation setup:

- Train/test split: 80:20.
- Cross validation: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Metrics: MAE, MSE, RMSE, and R2.
- Final selection metric: lowest CV RMSE.

Linear Regression scaling is fit only inside training folds during cross validation and only on the training set during final evaluation. Tree-based models do not require scaling.

## 5. Results

Cross validation results:

| Model combination | CV MAE | CV MSE | CV RMSE | CV R2 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest depth 14 | 11.9109 | 340.7061 | 18.4582 | 0.8469 |
| Decision Tree depth 16 | 13.2756 | 509.5903 | 22.5741 | 0.7710 |
| Decision Tree depth 12 | 15.2007 | 544.5843 | 23.3363 | 0.7553 |
| Decision Tree depth 8 | 19.5132 | 757.5012 | 27.5227 | 0.6595 |
| Linear Regression | 21.0904 | 821.1745 | 28.6561 | 0.6309 |

Final selected model: **Random Forest depth 14**

| Metric | Value |
| --- | ---: |
| Test MAE | 11.8832 |
| Test MSE | 348.7546 |
| Test RMSE | 18.6750 |
| Test R2 | 0.8449 |

Top features:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `arrival_date_week_number` | 0.2427 |
| 2 | `total_guests` | 0.2220 |
| 3 | `hotel_Resort Hotel` | 0.1071 |
| 4 | `lead_time` | 0.0494 |
| 5 | `arrival_date_year` | 0.0446 |

Output figures:

| Figure | Path |
| --- | --- |
| Model comparison | `output/figures/regression_model_comparison.png` |
| Actual vs predicted ADR | `output/figures/regression_actual_vs_predicted.png` |
| Residual analysis | `output/figures/regression_residual_analysis.png` |
| Feature importance | `output/figures/regression_feature_importance.png` |

Interpretation:

- The model's average absolute error is about 11.88 ADR units.
- RMSE is about 18.68 ADR units after stronger penalty for large errors.
- Test R2 of 0.8449 means the model explains about 84.49% of ADR variation on the test set.
- CV RMSE and test RMSE are close, so the selected model remains stable on unseen data.

## 6. Function Output Specification

`run_regression_modeling()` returns:

| Key | Description |
| --- | --- |
| `cv_result` | Cross validation results for all model/parameter combinations |
| `top5_result` | Top five combinations by CV RMSE |
| `test_result` | Final holdout test metrics for the selected model |
| `feature_importance` | Feature importance or absolute coefficient values |
| `best_model_name` | Name of the selected model |
| `best_model` | Fitted Scikit-learn estimator or pipeline |
| `feature_columns` | Final input feature names |
| `dropped_columns` | Columns excluded before modeling |

Usage:

```python
from src.regression import run_regression_modeling

result = run_regression_modeling()
print(result["best_model_name"])
print(result["top5_result"].round(4))
print(result["test_result"].round(4))
```

## 7. Stage Conclusion

The regression modeling stage is consistent with the proposal and project requirements. It uses encoded categorical variables, model-specific scaling, K-fold cross validation, and RMSE-based model selection. Random Forest depth 14 is the best model among the compared course-covered combinations and can support hotel pricing and revenue management decisions.

