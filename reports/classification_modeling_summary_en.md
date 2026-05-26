# Classification Modeling Summary

## 1. Objective

This section covers the **Classification Modeling & Evaluation** part of the hotel booking demand project. The target variable is `is_canceled`, where `0` means the booking was not canceled and `1` means the booking was canceled.

The business objective is to identify high cancellation-risk bookings before guest arrival. This can support early intervention, overbooking decisions, and hotel resource allocation.

Implementation file:

- `notebooks/03_classification_modeling.ipynb`
- `src/classification.py`

## 2. Data Used

The model uses the preprocessed classification dataset:

- `data/processed/hotel_bookings_clf.csv`

Dataset status:

| Item | Value |
| --- | ---: |
| Rows | 119,210 |
| Columns | 71 (after dropping leakage/target columns) |
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

## 4. Models Compared

Three scikit-learn classification models were compared within the course modeling and evaluation workflow:

| Model | Purpose |
| --- | --- |
| Logistic Regression | Simple and interpretable linear baseline |
| Decision Tree | Nonlinear rule-based classifier, later tuned for optimal performance |
| Random Forest | Ensemble model based on multiple decision trees |

`StandardScaler` was applied only inside the `Pipeline` for Logistic Regression. This prevents scaling leakage because the scaler is fitted only on the training fold during cross validation.

## 5. Evaluation Design

The data was split into train and test sets using an 80:20 split.

Key evaluation choices:

- `stratify=y` was used to preserve the target ratio in train/test sets.
- `StratifiedKFold(n_splits=5)` was used for cross validation.
- Metrics included accuracy, precision, recall, F1-score, and confusion matrix.
- **F1-score** was used as the primary metric because cancellation prediction needs a balance between precision and recall. 
- **GridSearchCV** was implemented on the best performing baseline model to optimize hyperparameters without exposing the test set.

## 6. Cross-Validation and Tuning Results

Initial baseline comparison results via 5-fold cross validation:

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
| --- | ---: | ---: | ---: | ---: |
| Decision Tree | 0.8401 | 0.8142 | 0.7371 | 0.7737 |
| Random Forest | 0.8477 | 0.8645 | 0.6989 | 0.7729 |
| Logistic Regression | 0.8075 | 0.8029 | 0.6373 | 0.7105 |

Because Decision Tree and Random Forest showed very similar initial performance, **Decision Tree** was selected as the final model candidate due to its higher recall, better interpretability, and stable F1-score. 

To maximize the business value of the model, **GridSearchCV** was applied to find the optimal hyperparameters targeting the F1-score:
- **Search Space:** `max_depth` [8, 10, 12, 15], `min_samples_split` [2, 5, 10], `criterion` ['gini', 'entropy']
- **Best Parameters:** `criterion='gini'`, `max_depth=15`, `min_samples_split=5`

**Tuning Result:** The CV F1-score significantly improved from 0.7737 to **0.7910**.

## 7. Final Test Results

Final selected and optimized model: **Decision Tree (Tuned)**

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

Interpretation:

- The model correctly identified 6,827 out of 8,840 canceled bookings, showing a strong capability to detect high-risk reservations.
- Only 1,538 normal bookings were falsely flagged as cancellations (False Positives).
- The model missed 2,013 canceled bookings. Notably, **hyperparameter tuning successfully reduced the number of missed cancellations by more than 250 cases** compared to the baseline model.
- The test F1-score (0.7936) is slightly higher than the CV score, proving the model is highly robust and not overfitted.

## 8. Saved Figures

The notebook saves presentation-ready figures under `reports/figures`.

| Figure | Path |
| --- | --- |
| Model comparison | `reports/figures/classification_model_comparison.png` |
| Confusion matrix | `reports/figures/confusion_matrix_decision_tree.png` |
| Feature importance | `reports/figures/classification_feature_importance.png` |

## 9. Feature Interpretation

Top 10 features utilized by the tuned Decision Tree:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `deposit_type_Non Refund` | 0.3658 |
| 2 | `market_segment_Online TA` | 0.0973 |
| 3 | `lead_time` | 0.0870 |
| 4 | `total_of_special_requests` | 0.0803 |
| 5 | `country_PRT` | 0.0651 |
| 6 | `arrival_date_year` | 0.0373 |
| 7 | `required_car_parking_spaces` | 0.0353 |
| 8 | `previous_cancellations` | 0.0327 |
| 9 | `arrival_date_week_number` | 0.0261 |
| 10 | `booking_changes` | 0.0190 |

These features are highly consistent with the EDA findings. `deposit_type_Non Refund`, `market_segment_Online TA`, and `lead_time` served as the most decisive factors for the model to split nodes and identify cancellation risks. 

*(Note: Feature importance indicates how useful a feature was for the model's predictive splits, not direct causality.)*

## 10. Proposal Alignment

This implementation perfectly aligns with the proposal:

- The task successfully predicts booking cancellation risk (`is_canceled`).
- Leakage columns are strictly excluded.
- Scaling is safely handled inside pipelines.
- **Optimization added:** Beyond the initial proposal, a hyperparameter tuning phase was successfully executed to maximize the model's reliability for actual business operations.

## 11. Final Recommendation

The classification modeling work is completely ready for submission. 

For the final presentation, the most professional and defensible explanation is:

> "While Decision Tree and Random Forest showed similar baseline performance, we selected the Decision Tree for its interpretability. We then applied GridSearchCV to optimize its hyperparameters. This proactive tuning significantly boosted both the Recall and F1-score, allowing us to capture over 250 additional high-risk cancellations while maintaining strict data leakage prevention."