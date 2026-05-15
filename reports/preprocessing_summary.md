# Preprocessing Summary

## Preprocessing Objective
The primary objective of this phase was to transform the raw hotel booking dataset into a clean, model-ready numerical format. This process focused on ensuring data integrity, preventing **Data Leakage**, and engineering features that provide logical insights for both cancellation (Classification) and ADR (Regression) prediction tasks.

---

## Data Cleaning & Intelligent Feature Conversion
We moved beyond simple cleaning by applying analytical logic to handle identifiers and outliers:
* **Binary Conversion of IDs (Agent/Company)**: Raw `agent` and `company` IDs were converted into binary flags (`has_agent`, `has_company`). This prevents machine learning models from incorrectly interpreting these nominal IDs as continuous numeric values with inherent order or importance.
* **Handling Missing Values**:
    * Missing `children` values were filled with 0.
    * Missing `country` values were replaced with 'Unknown'.
    * Following the creation of binary flags, the raw `agent` and `company` columns were dropped to reduce noise and missingness.
* **Strategic Outlier Filtering**:
    * **Regression Focus**: A strict range of $0 < ADR < 500$ was applied primarily to the **Regression dataset** to ensure target stability.
    * **Classification Consistency**: Outlier removal was handled carefully to ensure the distribution of cancellations remained representative of the original population.
    * **Invalid Bookings**: Entries with zero total guests (adults + children + babies) were purged.

---

## Feature Engineering & Leakage Prevention
Feature creation was driven by EDA insights, while leakage prevention ensured the validity of model evaluation:
* **New Feature Creation**:
    * `total_stay`: Calculated as the total sum of weekend and week nights to represent the stay duration.
    * `is_family`: A binary indicator to capture the specific behavior of family groups (bookings including children or babies).
* **Data Leakage Prevention**: The columns `reservation_status` and `reservation_status_date` were explicitly dropped. As these statuses are determined only after a booking is either fulfilled or canceled, including them would lead to artificial performance inflation and a model that cannot be used for real-time prediction.

---

## Categorical Encoding & Final Verification
To support mathematical modeling, all categorical variables were fully digitized:
* **Comprehensive One-Hot Encoding**: Applied to features including `hotel`, `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `arrival_date_month`, and notably **room types** (`reserved_room_type`, `assigned_room_type`) as requested.
* **Numeric Verification**: The final dataset was audited to ensure zero `object` or `string` columns remain, preventing potential execution errors in the modeling phase.

---

## Data Scaling Plan
In accordance with project requirements, **Feature Scaling** was intentionally omitted during this preprocessing phase. To avoid **Train/Test Leakage**, scaling will be implemented within the modeling pipeline, where the scaler will be fitted solely on the training data and then applied to the test data.

---

## Results Summary Table

| Metric | Raw Dataset | Classification (CLF) | Regression (REG) |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 119,390 | 117,396 | ~74,000* |
| **Total Columns** | 32 | ~66 | ~66 |
| **Missing Values** | 129,431+ | 0 | 0 |
| **Object Columns** | 12 | 0 | 0 |
*\*REG dataset includes only non-canceled bookings with strict ADR filtering.*

---

## Conclusion
The preprocessing stage successfully converted raw data into high-quality datasets. By addressing missing values through binary transformation, strictly isolating leakage variables, and separating datasets by prediction objective, this pipeline provides a robust and academically rigorous foundation for subsequent modeling.
