# Regression Modeling Validation Report

## 1. Validation Purpose

This document records the validation loop for the regression modeling work.

Reviewed implementation:

- `notebooks/04_regression_modeling.ipynb`
- `src/regression.py`
- `reports/regression_modeling_summary.md`
- `reports/regression_modeling_summary_en.md`
- `reports/regression_modeling_user_manual.md`
- `reports/figures/regression_model_comparison.png`
- `reports/figures/regression_actual_vs_predicted.png`
- `reports/figures/regression_residual_analysis.png`
- `reports/figures/regression_feature_importance.png`

## 2. Implementation Summary

The regression task predicts `adr` using the preprocessed regression dataset.

Key modeling choices:

| Area | Decision |
| --- | --- |
| Target | `adr` |
| Dataset | `data/processed/hotel_bookings_reg.csv` |
| Rows / columns | 73,386 rows x 83 columns |
| Final input features | 71 |
| Excluded columns | `adr`, `is_canceled`, `assigned_room_type_*` |
| CV method | 5-fold `KFold` |
| Metrics | MAE, MSE, RMSE, R² |
| Final model | Random Forest depth 14 |
| Open Source SW function | `run_regression_modeling()` in `src/regression.py` |

## 3. Final Metrics Verified

Cross-validation result for the selected model:

| Model | CV MAE | CV MSE | CV RMSE | CV R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest depth 14 | 11.9109 | 340.7130 | 18.4584 | 0.8469 |

Final test result:

| Model | Test MAE | Test MSE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Random Forest depth 14 | 11.8832 | 348.7546 | 18.6750 | 0.8449 |

The CV RMSE and test RMSE are close, so the selected model does not show a major test performance drop.

## 4. Local Verification

Local verification was run with the D-drive conda environment:

- Python environment: `D:/miniconda3/envs/ds`
- Notebook execution: successful
- Executed code cells: 12 / 12
- Notebook source Korean tokens: 0
- Each code cell begins with a short English comment.
- Top-level source function: `run_regression_modeling()`
- User manual: `reports/regression_modeling_user_manual.md`
- Function execution: successful with the D-drive `ds` conda environment
- Function result matched the notebook metrics: final model `Random Forest depth 14`, 71 input features, test RMSE 18.6750, test R² 0.8449

Generated figure files:

| Figure | Size |
| --- | --- |
| `regression_model_comparison.png` | 2631 x 1454 |
| `regression_actual_vs_predicted.png` | 2053 x 1753 |
| `regression_residual_analysis.png` | 3853 x 1454 |
| `regression_feature_importance.png` | 2360 x 1753 |

## 5. Agent Feedback Loop

Three validation agents reviewed the updated regression work.

| Agent | Focus | Result |
| --- | --- | --- |
| Agent A | Proposal alignment and leakage logic | No blocking issues. `adr` target, non-canceled dataset, ADR filtering, and feature exclusions are consistent with the proposal and preprocessing summary. |
| Agent B | Implementation and reproducibility | No blocking issues. End-to-end execution, figure generation, metrics, and leakage-safe scaling were verified. |
| Agent C | Documentation, presentation, and course-scope wording | No blocking issues. Notebook Markdown and Korean/English summaries are suitable for submission and presentation. |

Non-blocking feedback was reflected:

- The final model is described as the best among the five compared combinations, not as the global best possible model.
- The Korean summary wording for before-arrival ADR prediction was clarified.
- The English summary now includes clearer notes on assigned-room ablation, residual interpretation, and limited Random Forest parameter search.
- CV RMSE calculation is documented as `sqrt(mean CV MSE)`.
- The Open Source SW requirement is strengthened with a reusable top-level function and user manual.

## 6. Final Recommendation

The regression modeling work is ready for PR from modeling, documentation, and logic-consistency perspectives.

The safest presentation statement is:

> Random Forest depth 14 achieved the best ADR prediction performance among the five compared model/parameter combinations. It reached test MAE 11.88, RMSE 18.68, and R² 0.8449, which means it explains about 84.5% of ADR variation in the test set.

PR staging caution:

- Do not use `git add .`.
- Stage only the intended regression modeling files and generated regression figures.
