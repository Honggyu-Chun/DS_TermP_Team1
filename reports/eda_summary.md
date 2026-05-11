\# Exploratory Data Analysis (EDA) Summary



\## EDA Objective



This EDA was conducted to support two project objectives:



1\. Booking Cancellation Prediction (`is\_canceled`)

2\. ADR (Average Daily Rate) Prediction



The analysis focused on understanding booking patterns, identifying important variables, detecting missing values and outliers, and exploring relationships between features.



\---



\## Dataset Exploration



Basic dataset exploration included:



\* Dataset shape and sample records

\* Numerical and categorical variable classification

\* Binary variable identification

\* Missing value analysis

\* Statistical summary of key numerical variables



Missing counts and missing rates were examined to understand incomplete data.



\---



\## Cancellation Analysis



Several visualizations were created to analyze cancellation behavior:



\* Distribution of cancellation status

\* Cancellation rate by hotel type

\* Cancellation rate by deposit type

\* Cancellation rate by market segment

\* Cancellation rate by customer type

\* Cancellation rate by lead time group

\* Cancellation rate by total special requests



The analysis showed that cancellation patterns vary depending on customer and booking characteristics.



\---



\## ADR Analysis



ADR-related analysis included:



\* Histogram of ADR

\* Boxplot of ADR

\* ADR distribution by hotel type

\* Average ADR by arrival month

\* Scatter plot of lead time and ADR



The results indicated that ADR has a right-skewed distribution with several outliers and seasonal patterns.



\---



\## Outlier and Correlation Analysis



Outlier analysis identified:



\* Extremely high ADR values

\* ADR values less than or equal to zero

\* Extremely large lead time values

\* Bookings with zero guests



A correlation matrix was also generated to examine relationships between important variables such as `is\_canceled`, `adr`, `lead\_time`, and `previous\_cancellations`.



\---



\## Conclusion



The EDA provided useful insights for both cancellation prediction and ADR prediction tasks. The findings from this analysis will support preprocessing, feature engineering, and machine learning model development.



