# Preprocessing Summary

## 1. Preprocessing Objective
The primary objective of this phase was to transform the raw hotel booking dataset into a clean, fully digitized, and model-ready numerical format. Beyond simple cleaning, this phase focused on establishing a rigorous pipeline to prevent **Data Leakage** and manage **Dimensionality Explosion** for categorical variables, ensuring high model reliability for both Classification (Cancellation) and ADR (Regression) prediction tasks.

---

## 2. Data Cleaning & Advanced Filtering
To ensure data integrity, the following analytical filters and cleaning methods were applied:
* **Column Sanitization**: All column names were stripped of leading/trailing whitespaces upon loading to prevent data call errors and ensure stable processing.
* **Removal of "Ghost Bookings"**: Purged **180 invalid entries** where the sum of adults, children, and babies was zero, as these represent non-existent guest profiles. *Based on EDA, these were identified as invalid or corrupted reservations and removed. As a result, the raw dataset dropped from 119,390 rows to 119,210 rows for the classification task.*
* **Strategic Imputation**:
    * Missing `children` values were filled with 0.
    * Missing `country` values were replaced with 'Unknown'.
* **Binary Conversion of IDs**: Raw `agent` and `company` IDs were converted into binary flags (`has_agent`, `has_company`) and the original columns were dropped. This prevents models from incorrectly treating nominal IDs as continuous numeric scales.
* **Target-Specific Filtering**:
    * **Regression Focus**: A strict range of $0 < ADR < 500$ was applied to the **Regression dataset** to ensure target stability and remove extreme outliers. *Regression dataset contains only non-canceled bookings because ADR prediction is intended for completed reservations. Therefore, `is_canceled` is constant (0) within this dataset and will be excluded from regression input features to avoid zero-variance redundancy.*

---

## 3. Dimensionality Management (Country Data)
The raw dataset contained hundreds of unique country codes. To prevent a feature explosion during encoding, which leads to a sparse matrix and potential overfitting, we implemented a consolidation strategy:
* **Top 10 Grouping**: Only the **Top 10 countries** by frequency were retained as unique features.
* **Residual Categorization**: All other countries were grouped into a single 'Other' category, significantly optimizing the One-Hot Encoding process and enhancing model generalization.

---

## 4. Feature Engineering & Leakage Prevention
* **New Feature Creation**:
    * `total_stay`: Sum of weekend and week nights to represent actual stay duration.
    * `is_family`: A binary indicator identifying bookings that include children or babies.
* **Data Leakage Prevention**: The columns `reservation_status` and `reservation_status_date` were explicitly dropped. Since these are determined only after a stay is completed or canceled, their inclusion would cause artificial performance inflation and render the model unusable for real-time prediction.
* **Data Scaling Strategy**: *Data scaling was intentionally bypassed during this preprocessing phase to prevent train/test data leakage. Global scaling before dividing the data causes leakage. Instead, scaling will be applied within the modeling pipeline after the train/test split (fitting on the train set only, and transforming both splits).*
* **Assigned Room Type Risk Management**: *Since `assigned_room_type` is determined at check-in rather than booking time, it introduces target leakage risks. We assume assigned room type availability before prediction for validation purposes, but we will perform a feature-ablation experiment during modeling to evaluate its predictive stability compared to `reserved_room_type`.*

---

## 5. Categorical Encoding & Final Audit
* **Comprehensive One-Hot Encoding**: Applied to `hotel`, `meal`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `arrival_date_month`, and notably **all room types** (`reserved_room_type`, `assigned_room_type`).
* **Numeric Verification**: The final dataset was audited to ensure exactly **zero `object` or `string` columns** remain, resulting in a finalized count of **83 numeric columns including the target variable**.

---

## 6. Results Summary Table

| Metric | Raw Dataset | Classification (CLF) | Regression (REG) |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 119,390 | **119,210** | **73,386*** |
| **Total Columns** | 32 | **83** | **83** |
| **Missing Values** | 129,431+ | 0 | 0 |
| **Object Columns** | 12 | 0 | 0 |
*\*The REG dataset consists exclusively of non-canceled bookings with strict ADR filtering ($0 < ADR < 500$), aligning exactly with the final audited count of 73,386 rows.*

---

## 7. Conclusion
Through this preprocessing pipeline, we successfully cleaned the data noise and expanded the feature set from 32 to **83 fully digitized columns including the target variable**. By synchronizing the logic between the Python script and the Jupyter notebook, we have established a reproducible and academically rigorous foundation for the subsequent modeling phase.