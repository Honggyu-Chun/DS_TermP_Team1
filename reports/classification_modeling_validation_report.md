# Classification Modeling Validation Report

## 1. Validation Purpose

This document records the A+ refinement and validation loop for the classification modeling work.

Reviewed implementation:

- `notebooks/03_classification_modeling.ipynb`
- `src/classification.py`
- `reports/classification_modeling_summary.md`
- `reports/classification_modeling_summary_en.md`
- `reports/figures/classification_model_comparison.png`
- `reports/figures/confusion_matrix_decision_tree.png`
- `reports/figures/classification_feature_importance.png`

## 2. Improvements Applied

The following improvements were applied after the external review feedback and subsequent tuning phase.

| Area | Improvement |
| --- | --- |
| Hyperparameter Tuning | Applied `GridSearchCV` to optimize the Decision Tree model, improving the final test F1-score from 0.7749 to 0.7936. |
| Notebook submission language | Notebook Markdown was changed to English for final submission readiness. |
| Model selection explanation | Decision Tree is no longer described as strongly superior. The explanation now says Decision Tree and Random Forest are very close, but Decision Tree was selected due to slightly higher F1-score, higher recall, and better interpretability. |
| Presentation figures | Model comparison, confusion matrix, and feature importance plots are saved under `reports/figures` reflecting tuned results. |
| English report | `classification_modeling_summary_en.md` was added for English submission/report integration. |
| Korean presenter report | The Korean summary was updated with safer presentation wording, tuning results, and figure paths. |
| KNN explanation | KNN was considered but not used in the full-data 5-fold comparison because of high distance-computation cost on 119,210 rows. |
| SMOTE explanation | SMOTE was not applied because the target ratio is 62.92% vs 37.08%, which is not an extreme imbalance for this baseline. |

## 3. Local Verification

Local verification was run with the D-drive conda environment:

- Python environment: Local Python Environment (Python 3.10)
- Notebook execution: successful
- Executed code cells: 11 / 11
- Saved figures: generated successfully

Generated figure files:

| Figure | Status |
| --- | --- |
| `classification_model_comparison.png` | Regenerated successfully |
| `confusion_matrix_decision_tree.png` | Regenerated successfully (Tuned) |
| `classification_feature_importance.png` | Regenerated successfully (Tuned) |

## 4. Final Metrics Verified

Cross-validation result (Baseline):

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8401 | 0.8142 | 0.7371 | 0.7737 |
| Random Forest | 0.8477 | 0.8645 | 0.6989 | 0.7729 |
| Logistic Regression | 0.8075 | 0.8029 | 0.6373 | 0.7105 |

Final test result (After GridSearchCV Tuning):

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree (Tuned) | 0.8511 | 0.8161 | 0.7723 | 0.7936 |

Confusion matrix:

| Actual / Predicted | Predicted not canceled | Predicted canceled |
| --- | ---: | ---: |
| Actual not canceled | 13,464 | 1,538 |
| Actual canceled | 2,013 | 6,827 |

## 5. Agent Feedback Loop

Three validation agents reviewed the updated work.

| Agent | Focus | Result |
| --- | --- | --- |
| Agent A | Proposal alignment and leakage logic | No blocking issues. Target, leakage handling, feature exclusions, and model selection explanation are logically consistent. |
| Agent B | Implementation and reproducibility | No blocking issues. Notebook execution, saved figures, metrics, and leakage-safe scaling were verified. |
| Agent C | Documentation, presentation, and course-scope wording | No blocking issues. English notebook Markdown and Korean/English reports are suitable. Suggested minor wording improvements were applied. |

## 6. Final Recommendation

The classification modeling work is ready for PR from the modeling, documentation, and logic-consistency perspectives.

The safest presentation statement is:

> Decision Tree and Random Forest showed very similar baseline performance. We selected Decision Tree for its interpretability, and further applied GridSearchCV to optimize its parameters. This successfully improved our test F1-score to 0.7936 and recall to 0.7723, providing a strong and defensible model for business explanation.

PR staging caution:

- Do not use `git add .`.
- Stage only the intended modeling files (`.ipynb`, `.py`), summary markdown files, and generated classification figures.