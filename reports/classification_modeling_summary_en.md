# Classification Modeling Summary

## 1. Objective

This section covers the **Classification Modeling & Evaluation** part of the hotel booking demand project. The target variable is `is_canceled`, where `0` means the booking was not canceled and `1` means the booking was canceled.

The business objective is to identify high cancellation-risk bookings before guest arrival. This can support early intervention, overbooking decisions, and hotel resource allocation.

Implementation file:

- `notebooks/03_classification_modeling.ipynb`

## 2. Data Used

The model uses the preprocessed classification dataset:

- `data/processed/hotel_bookings_clf.csv`

Dataset status:

| Item | Value |
| --- | ---: |
| Rows | 119,210 |
| Columns | 83 |
| Missing values | 0 |
| Object/string columns | 0 |
| Target | `is_canceled` |
| Not canceled `0` | 75,011, 62.92% |
| Canceled `1` | 44,199, 37.08% |

The preprocessing stage already handled missing values, removed invalid zero-guest bookings, created derived features, applied one-hot encoding, and removed direct leakage columns such as `reservation_status` and `reservation_status_date`.

## 3. Feature Selection Logic

The following columns were excluded from model input `X`.

| Excluded columns | Reason |
| --- | --- |
| `is_canceled` | This is the target variable. |
| `adr` | This is the regression target for the second modeling task, so it was removed to keep classification and regression tasks separated. |
| `assigned_room_type_*` | Assigned room type may not be available at booking-time prediction, so it was removed to reduce prediction-time leakage risk. |

After these exclusions, the classification model uses **71 input features**.

This is a conservative and defensible decision because the model is intended for early cancellation risk prediction. `reserved_room_type_*` can be known at reservation time, while `assigned_room_type_*` can be determined later by hotel operations.

As a future ablation study, the team can compare models with and without `assigned_room_type_*`. This would show how much prediction-time room assignment information changes model performance.

## 4. Models Compared

Three scikit-learn classification models were compared within the course modeling and evaluation workflow:

| Model | Purpose |
| --- | --- |
| Logistic Regression | Simple and interpretable linear baseline |
| Decision Tree | Nonlinear rule-based classifier |
| Random Forest | Ensemble model based on multiple decision trees |

`StandardScaler` was applied only inside the `Pipeline` for Logistic Regression. This prevents scaling leakage because the scaler is fitted only on the training fold during cross validation.

KNN was considered but not included in the main full-data comparison because distance-based 5-fold cross validation on 119,210 rows can be computationally expensive. The selected model set still covers a simple interpretable baseline, a tree model, and an ensemble model.

## 5. Evaluation Design

The data was split into train and test sets using an 80:20 split.

Key evaluation choices:

- `stratify=y` was used to preserve the target ratio in train/test sets.
- `StratifiedKFold(n_splits=5)` was used for cross validation.
- Metrics included accuracy, precision, recall, F1-score, and confusion matrix.
- The final model was selected by cross-validation F1-score.

F1-score was used because cancellation prediction needs a balance between precision and recall. Missing too many actual cancellations is costly, but flagging too many normal bookings also creates unnecessary operational work.

SMOTE was not applied in this baseline because the target distribution is about 62.92% not canceled and 37.08% canceled, which is not an extreme class imbalance. If the next modeling goal is to increase recall further, class weighting or SMOTE can be tested as an additional experiment.

## 6. Cross-Validation Results

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8401 | 0.8142 | 0.7371 | 0.7737 |
| Random Forest | 0.8477 | 0.8645 | 0.6989 | 0.7729 |
| Logistic Regression | 0.8075 | 0.8029 | 0.6373 | 0.7105 |

Decision Tree and Random Forest performed very similarly. The F1-score difference is only `0.0008`, so Decision Tree should not be described as overwhelmingly better. Decision Tree was selected because it had slightly higher F1-score, higher recall, and better interpretability for presentation.

## 7. Final Test Results

Final selected model:

- Decision Tree

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8399 |
| Precision | 0.8095 |
| Recall | 0.7431 |
| F1-score | 0.7749 |

Confusion matrix:

| Actual / Predicted | Predicted not canceled | Predicted canceled |
| --- | ---: | ---: |
| Actual not canceled | 13,456 | 1,546 |
| Actual canceled | 2,271 | 6,569 |

Interpretation:

- The model correctly identified 6,569 out of 8,840 canceled bookings.
- The model missed 2,271 canceled bookings.
- The test F1-score is close to the cross-validation F1-score, so the result does not look heavily overfitted.

## 8. Saved Figures

The notebook saves presentation-ready figures under `reports/figures`.

| Figure | Path |
| --- | --- |
| Model comparison | `reports/figures/classification_model_comparison.png` |
| Confusion matrix | `reports/figures/confusion_matrix_decision_tree.png` |
| Feature importance | `reports/figures/classification_feature_importance.png` |

These figures can be used directly in the final report or presentation slides.

## 9. Feature Interpretation

Top Decision Tree features included:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `deposit_type_Non Refund` | 0.4300 |
| 2 | `market_segment_Online TA` | 0.1117 |
| 3 | `total_of_special_requests` | 0.0896 |
| 4 | `country_PRT` | 0.0759 |
| 5 | `lead_time` | 0.0750 |

These features are consistent with the EDA findings because cancellation patterns differed by deposit type, market segment, special requests, and lead time.

Feature importance is not causal evidence. It only means the trained model used those variables frequently or strongly for splitting.

## 10. Proposal Alignment

This implementation aligns with the proposal:

- The target is `is_canceled`.
- The task predicts booking cancellation risk.
- The model uses preprocessed numerical data with categorical encoding.
- Leakage columns are excluded.
- Scaling is handled in the modeling pipeline, not globally before splitting.
- Classification evaluation uses k-fold cross validation and multiple metrics.

## 11. Final Recommendation

The classification modeling work is ready as a strong baseline. Decision Tree is a defensible final model because it has slightly stronger recall and F1-score than Random Forest while being easier to explain.

For presentation, the safest explanation is:

> Decision Tree and Random Forest had very similar performance. We selected Decision Tree because it had slightly better recall and F1-score, and its decision logic is easier to interpret for business explanation.
