Data Preprocessing & Feature Engineering Summary
 Objectives
The primary goal of this preprocessing phase is to clean, transform, and optimize the raw hotel booking data for machine learning tasks, specifically Booking Cancellation Prediction (Classification) and ADR Prediction (Regression).

 Data Cleaning
To ensure model stability and accuracy, noise and missing values were handled based on insights from the EDA.

Handling Missing Values:

children: Replaced missing values with 0.

country: Replaced missing values with 'Unknown'.

agent: Replaced missing values with 0 (representing no agent).

company: Column was removed due to a high missing rate (>94%).

Outlier Removal:

ADR (Average Daily Rate): Filtered to keep values between 0 and 500 to remove extreme outliers and erroneous entries.

Zero Guests: Removed records where the total number of guests (Adults + Children + Babies) was 0, as these represent invalid bookings.

 Feature Engineering
New features were engineered to capture underlying patterns and improve predictive power.

Derived Features:

total_stay: Combined weekend and weeknight stays into a single "Total Stay Duration" feature.

is_family: A binary feature indicating if the booking includes children or babies.

Feature Simplification:

country: High-cardinality country data was simplified by keeping the Top 10 countries and grouping the rest into an 'Other' category.

Data Leakage Prevention:

Removed reservation_status and reservation_status_date. These features are updated only after a cancellation or check-out occurs, and including them would lead to unrealistic model performance.

 Encoding
Categorical data was converted into a numerical format suitable for machine learning algorithms.

One-Hot Encoding: Applied to categorical variables such as hotel, meal, market_segment, and country.

Result: The feature space was expanded to 66 columns, providing the model with distinct indicators for each category.

 Final Deliverables
Two specialized datasets were exported for subsequent modeling roles:

hotel_bookings_clf.csv: Prepared for Role #4 (Classification). Includes all valid records for predicting is_canceled.

hotel_bookings_reg.csv: Prepared for Role #5 (Regression). Includes only non-canceled bookings (where is_canceled == 0) to predict the actual ADR.

 Quality Assurance
Final Null Count: 0 (Perfectly cleaned).

Correlation Analysis: Confirmed that features like lead_time, deposit_type, and total_of_special_requests maintain significant relationships with the target variable after processing.