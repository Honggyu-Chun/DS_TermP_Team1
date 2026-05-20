# Regression Modeling Summary

## 1. Objective

This section covers the **ADR regression modeling** part of the hotel booking demand project. The target variable is `adr` (Average Daily Rate), a continuous numerical value.

The business objective is to support hotel revenue management and pricing decisions by estimating ADR from booking conditions such as hotel type, stay length, number of guests, market segment, room type, and seasonality variables.

Implementation file:

- `notebooks/04_regression_modeling.ipynb`

## 2. Data Used

The model uses the preprocessed regression dataset:

- `data/processed/hotel_bookings_reg.csv`

Dataset status:

| Item | Value |
| --- | ---: |
| Rows | 73,386 |
| Columns | 83 |
| Missing values | 0 |
| Object/string columns | 0 |
| Target | `adr` |
| `is_canceled` | Constant 0 |
| ADR mean | 102.373 |
| ADR median | 94.500 |
| ADR std | 47.222 |
| ADR min | 0.260 |
| ADR max | 451.500 |

The regression dataset contains only non-canceled bookings and applies the ADR filter `0 < adr < 500`. This follows the preprocessing logic that ADR prediction should focus on completed reservations and avoid extreme target outliers.

## 3. Feature Selection Logic

The following columns were removed from model input `X`.

| Excluded columns | Reason |
| --- | --- |
| `adr` | This is the regression target. |
| `is_canceled` | The regression dataset contains only non-canceled bookings, so this column is constant 0. |
| `assigned_room_type_*` | Assigned room type may not be available at booking-time prediction, so it is removed to reduce prediction-time leakage risk. |

After these exclusions, the regression model uses **71 input features**.

`reserved_room_type_*` is kept because it is known at reservation time. `assigned_room_type_*` is excluded because it may be determined later by hotel operations.

This is a conservative choice for before-arrival ADR prediction. A future ablation study can compare model performance with and without `assigned_room_type_*`.

## 4. Models and Evaluation

Five model/parameter combinations were compared:

| Model combination | Purpose |
| --- | --- |
| Linear Regression | Simple continuous prediction baseline |
| Decision Tree depth 8 | Shallow nonlinear tree model |
| Decision Tree depth 12 | Medium-depth tree model |
| Decision Tree depth 16 | More complex tree model |
| Random Forest depth 14 | Ensemble regression model |

Evaluation design:

- `KFold(n_splits=5, shuffle=True, random_state=42)` was used for cross validation.
- MAE, MSE, RMSE, and R² were calculated.
- The final model was selected by the lowest cross-validation RMSE.

RMSE was used as the main selection metric because large ADR errors can be more harmful for pricing and revenue decisions.

In this notebook, CV RMSE is calculated as `sqrt(mean CV MSE)`. The reported CV RMSE values are consistent with that calculation.

## 5. Cross-Validation Results

| Model | CV MAE | CV MSE | CV RMSE | CV R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest depth 14 | 11.9109 | 340.7130 | 18.4584 | 0.8469 |
| Decision Tree depth 16 | 13.2756 | 509.5903 | 22.5741 | 0.7710 |
| Decision Tree depth 12 | 15.2007 | 544.5843 | 23.3363 | 0.7553 |
| Decision Tree depth 8 | 19.5132 | 757.5012 | 27.5227 | 0.6595 |
| Linear Regression | 21.0904 | 821.1745 | 28.6561 | 0.6309 |

Among the five compared model/parameter combinations, Random Forest depth 14 had the lowest CV RMSE and the highest CV R². This suggests that ADR has nonlinear patterns that are better captured by tree-based ensemble learning than by a simple linear baseline.

## 6. Final Test Results

Final selected model:

- Random Forest depth 14

| Metric | Value |
| --- | ---: |
| Test MAE | 11.8832 |
| Test MSE | 348.7546 |
| Test RMSE | 18.6750 |
| Test R² | 0.8449 |

Interpretation:

- The average absolute ADR error is about 11.88.
- RMSE is about 18.68, which penalizes larger ADR errors.
- R² is 0.8449, meaning the model explains about 84.49% of ADR variation in the test set.
- CV RMSE 18.4584 and test RMSE 18.6750 are close, so the model does not show a major test performance drop.

This means Random Forest depth 14 is the best model among the tested combinations, not that every possible regression model and hyperparameter setting was exhaustively searched.

## 7. Saved Figures

The notebook saves presentation-ready figures under `reports/figures`.

| Figure | Path |
| --- | --- |
| Model comparison | `reports/figures/regression_model_comparison.png` |
| Actual vs predicted ADR | `reports/figures/regression_actual_vs_predicted.png` |
| Residual analysis | `reports/figures/regression_residual_analysis.png` |
| Feature importance | `reports/figures/regression_feature_importance.png` |

These figures can be used directly in the final report or presentation slides.

The residual analysis figure is used to check whether prediction errors are centered around zero and whether error patterns become larger in certain predicted ADR ranges.

## 8. Feature Interpretation

Top Random Forest features included:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `arrival_date_week_number` | 0.2427 |
| 2 | `total_guests` | 0.2220 |
| 3 | `hotel_Resort Hotel` | 0.1071 |
| 4 | `lead_time` | 0.0494 |
| 5 | `arrival_date_year` | 0.0446 |

These features align with the proposal because ADR prediction was expected to depend on seasonality, hotel type, number of guests, market segment, and room type.

Feature importance is not causal evidence. It only shows which variables the trained model used most strongly for prediction.

## 9. Proposal Alignment

This implementation aligns with the proposal:

- The regression target is `adr`.
- The model supports revenue management and pricing decisions.
- Candidate features such as hotel type, guests, market segment, room type, stay information, and seasonality are included through preprocessed numerical features.
- Missing values and categorical encoding were handled in preprocessing.
- Scaling is applied only inside the Linear Regression pipeline.
- Evaluation uses MAE, MSE, RMSE, and R².

## 10. Limitations and Next Steps

The current model is a strong regression baseline. Future improvements can include:

- Feature ablation with and without `assigned_room_type_*`.
- Time-based split for stricter future-booking evaluation.
- More Random Forest parameter combinations. The current parameter search is intentionally limited for a clear baseline.
- More detailed residual analysis by ADR range or hotel type.

## 11. Final Recommendation

The regression modeling work is ready as a strong baseline. Random Forest depth 14 is the best current model among the five compared combinations because it has the lowest CV RMSE, the lowest test error, and the highest R².

Safe presentation statement:

> Random Forest depth 14 achieved the best ADR prediction performance among the five compared model/parameter combinations. It reached test MAE 11.88, RMSE 18.68, and R² 0.8449, which means it explains about 84.5% of ADR variation in the test set.
