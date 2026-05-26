# Exploratory Data Analysis (EDA) Summary

## 1. Purpose

The purpose of the EDA stage is to understand the Kaggle Hotel Booking Demand dataset before preprocessing and modeling. This stage supports two project targets:

- Classification target: `is_canceled`
- Regression target: `adr`

The EDA identifies data quality issues, important booking patterns, potential outliers, and relationships that guide feature engineering and modeling decisions.

## 2. Reason

EDA is required because the raw dataset contains mixed numerical and categorical variables, missing values, outliers, and booking information that may create leakage if used without review. The project also has two different modeling goals, so EDA must explain both cancellation behavior and ADR variation.

Business reasons:

- Cancellation analysis helps hotels understand which reservations are more likely to be canceled.
- ADR analysis helps hotels understand price patterns across seasonality, hotel type, and booking conditions.
- Data quality analysis prevents invalid records from weakening model reliability.

## 3. Logic

The analysis follows this logic:

1. Inspect the raw dataset structure and variable types.
2. Identify missing values and abnormal records before modeling.
3. Analyze cancellation patterns by important booking features.
4. Analyze ADR distribution, outliers, and seasonal patterns.
5. Use the findings to justify preprocessing, feature engineering, and model evaluation choices.

This logic connects EDA directly to the later preprocessing, classification, and regression stages.

## 4. Method

The EDA includes:

- Dataset shape and sample record inspection.
- Numerical, categorical, and binary variable classification.
- Missing value count and missing rate review.
- Statistical summaries of key numerical variables.
- Cancellation plots by hotel type, deposit type, market segment, customer type, lead time group, and special requests.
- ADR histograms, boxplots, hotel/month comparisons, and lead time vs ADR scatter plots.
- Outlier checks for high ADR values, non-positive ADR values, large lead time values, and zero-guest bookings.
- Correlation matrix review for important numerical variables.

Key figures are saved under `reports/figures`.

## 5. Results

Main EDA findings:

- Cancellation rates vary meaningfully by deposit type, market segment, lead time, customer type, and total special requests.
- ADR has a right-skewed distribution with clear outliers.
- ADR patterns differ by hotel type and arrival month, indicating seasonal and operational price effects.
- Zero-guest bookings exist and should be removed during preprocessing.
- Missing values in `children` and `country` require explicit handling.
- `reservation_status` and `reservation_status_date` are leakage-prone because they are determined after the booking outcome.

These findings justify the later preprocessing decisions: missing value handling, zero-guest filtering, ADR filtering for regression, one-hot encoding, feature engineering, and leakage prevention.

## 6. Stage Conclusion

The EDA stage provides the evidence base for the entire project pipeline. It confirms that the project should use encoded categorical features, leakage-safe preprocessing, cancellation classification, and ADR regression.

