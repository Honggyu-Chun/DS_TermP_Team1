# Preprocessing Summary

## Preprocessing Objective
This preprocessing phase was conducted to transform the raw dataset into a clean, numerical format ready for machine learning. The primary goals were to handle missing values, remove outliers, prevent data leakage, and engineer new features to improve model performance for both cancellation and ADR prediction.

---

## Data Cleaning & Outlier Removal
Specific cleaning steps were implemented to ensure data integrity:
* **Column Trimming**: Whitespace was removed from column names to prevent indexing errors.
* **Handling Missing Values**:
    * Missing `children` values were filled with 0.
    * Missing `country` values were replaced with 'Unknown'.
    * Missing `agent` values were filled with 0.
    * The `company` column was dropped due to excessive missing data.
* **Outlier Filtering**:
    * ADR values were restricted to the range of 0 < ADR < 500.
    * Bookings with zero total guests (adults + children + babies) were removed from the dataset.

---

## Feature Engineering & Leakage Prevention
New variables were created to better capture booking behavior, and sensitive columns were removed:
* **New Feature Creation**:
    * `total_stay`: Calculated as the sum of weekend and week nights.
    * `is_family`: A binary indicator for bookings including children or babies.
* **Category Simplification**: The `country` variable was simplified by keeping only the top 10 countries and grouping the rest as 'Other'.
* **Data Leakage Prevention**: The columns `reservation_status` and `reservation_status_date` were dropped because they are determined after the cancellation outcome is known.

---

## Categorical Encoding & Verification
Categorical data was converted into a format suitable for mathematical modeling:
* **One-Hot Encoding**: Applied to categorical features such as `hotel`, `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `arrival_date_month`, and `country`.
* **Verification**:
    * Confirmed zero missing values in the final dataset.
    * Analyzed feature correlations with `is_canceled` to identify the most influential predictors.
* The final processed dataset contains 66 columns.

---

## Data Export
The processed data was saved into two distinct files to support the project's secondary objectives:
* **hotel_bookings_clf.csv**: The full processed dataset for Classification (Cancellation Prediction).
* **hotel_bookings_reg.csv**: A subset containing only non-canceled bookings for Regression (ADR Prediction).

---

## Conclusion
The preprocessing stage successfully converted raw data into high-quality, model-ready datasets. By addressing missing values, outliers, and data leakage, this pipeline provides a solid foundation for the subsequent modeling and evaluation phases.