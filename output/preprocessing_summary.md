# Preprocessing Summary

## 1. Purpose

The purpose of the preprocessing stage is to transform the raw Hotel Booking Demand dataset into clean, numerical, model-ready datasets for both classification and regression.

Outputs:

- Classification dataset: `data/processed/hotel_bookings_clf.csv`
- Regression dataset: `data/processed/hotel_bookings_reg.csv`

Implementation file:

```text
src/processing.py
```

Top-level function:

```python
preprocess_hotel_booking_data(
    raw_data_path="data/raw/hotel_bookings.csv",
    output_dir="data/processed",
)
```

## 2. Reason

Preprocessing is necessary because the raw dataset contains missing values, categorical variables, invalid zero-guest records, high-cardinality country values, leakage-prone status columns, and ADR outliers.

Project reasons:

- Classification requires a clean dataset to predict `is_canceled`.
- Regression requires a clean non-canceled booking dataset to predict `adr`.
- Scikit-learn models require numerical input, so categorical variables must be encoded.
- Global scaling is avoided at this stage to prevent train/test leakage; scaling is applied later inside modeling pipelines where needed.

## 3. Logic

The preprocessing logic is designed to preserve reproducibility and prevent leakage:

1. Load the raw CSV and strip column name whitespace.
2. Fill missing `children` values with 0 before guest-count validation.
3. Fill missing `country` values with `Unknown`.
4. Calculate `total_guests`.
5. Remove zero-guest bookings after missing guest counts are resolved.
6. Convert `agent` and `company` IDs into binary flags (`has_agent`, `has_company`).
7. Create booking-level derived features.
8. Reduce country cardinality by keeping the top 10 countries and grouping the rest as `Other`.
9. Drop direct leakage columns (`reservation_status`, `reservation_status_date`).
10. One-hot encode categorical variables.
11. Export separate classification and regression datasets.

This order is important. In particular, `children` missing values must be filled before `total_guests` is computed so that incomplete child-count records are not incorrectly removed as invalid bookings.

## 4. Method

Cleaning and feature engineering:

- `children`: missing values filled with 0.
- `country`: missing values filled with `Unknown`, then grouped into top 10 countries or `Other`.
- `total_guests`: `adults + children + babies`.
- `total_stay`: `stays_in_weekend_nights + stays_in_week_nights`.
- `is_family`: 1 when children or babies are present, otherwise 0.
- `has_agent`: 1 when an agent ID exists, otherwise 0.
- `has_company`: 1 when a company ID exists, otherwise 0.

Note: `total_stay` corresponds to `total_stay_days` in the project proposal. The existing name is kept for notebook and report compatibility.

Encoding:

- One-hot encoding is applied to `hotel`, `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `arrival_date_month`, `country`, `reserved_room_type`, and `assigned_room_type`.
- Boolean dummy columns are converted to integers for cross-platform consistency.

Dataset split:

- Classification dataset keeps all valid bookings.
- Regression dataset keeps only non-canceled bookings and applies `0 < adr < 500`.

## 5. Results

Final audit:

| Metric | Raw Dataset | Classification Dataset | Regression Dataset |
| --- | ---: | ---: | ---: |
| Rows | 119,390 | 119,210 | 73,386 |
| Columns | 32 | 83 | 83 |
| Missing values after processing | N/A | 0 | 0 |
| Object columns after processing | N/A | 0 | 0 |

The preprocessing function returns a dictionary with:

| Key | Description |
| --- | --- |
| `classification_path` | Saved classification CSV path |
| `regression_path` | Saved regression CSV path |
| `raw_shape` | Raw dataset shape |
| `classification_shape` | Classification dataset shape |
| `regression_shape` | Regression dataset shape |
| `missing_values_after_processing` | Remaining missing values in the classification dataset |
| `object_columns_after_processing` | Remaining object columns in the classification dataset |

## 6. Stage Conclusion

The preprocessing stage produces reproducible, fully numerical, leakage-aware datasets for modeling. It satisfies the project requirement for categorical encoding and supports later scaling through model-specific Scikit-learn pipelines rather than global preprocessing.

