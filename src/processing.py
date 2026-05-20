# src/processing.py
import os
import pandas as pd
import numpy as np

def preprocess_hotel_booking_data(raw_data_path: str, output_dir: str):
    """
    Executes the end-to-end hotel booking data preprocessing pipeline.
    Synchronizes logic between script and notebook, ensuring no absolute paths are leaked.
    """
    print("=" * 60)
    print("STARTING HOTEL BOOKING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # 1. Raw Data Loading & Validation
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at target path: {raw_data_path}")
        
    df_raw = pd.read_csv(raw_data_path)
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    print(f"Loaded raw dataset successfully. Initial Shape: {df.shape}")
    
    # 2. Data Cleaning & Anomaly Resolution (Feedback #1)
    # Missing values must be resolved before zero-guest filtering so unknown
    # children counts are not treated as invalid bookings.
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')

    # Filter out invalid "Ghost Bookings" where total guest count is zero
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df = df[df['total_guests'] > 0]
    print(f"Filtered zero-guest booking anomalies. Remaining records: {len(df)}")
    
    # Convert IDs into binary flags to prevent models from treating nominal IDs as continuous scales
    if 'agent' in df.columns:
        df['has_agent'] = df['agent'].notnull().astype(int)
        df.drop('agent', axis=1, inplace=True)
    if 'company' in df.columns:
        df['has_company'] = df['company'].notnull().astype(int)
        df.drop('company', axis=1, inplace=True)
        
    # 3. Feature Engineering & Dimensionality Management
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
    
    # Consolidate high cardinality country data to prevent sparse matrix explosion
    top_10_countries = df['country'].value_counts().nlargest(10).index
    df['country'] = df['country'].apply(lambda x: x if x in top_10_countries else 'Other')
    
    # 4. Data Leakage Prevention
    # Drop columns that are determined after the booking outcome
    leak_cols = ['reservation_status', 'reservation_status_date']
    df.drop([c for c in leak_cols if c in df.columns], axis=1, inplace=True)
    
    # 5. Categorical One-Hot Encoding (Including Room Types - Feedback #5)
    cat_cols = [
        'hotel', 'meal', 'market_segment', 'distribution_channel', 
        'deposit_type', 'customer_type', 'arrival_date_month', 'country',
        'reserved_room_type', 'assigned_room_type'
    ]
    df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Convert bool types to int for cross-platform model alignment
    bool_cols = df_final.select_dtypes(include=['bool']).columns
    df_final[bool_cols] = df_final[bool_cols].astype(int)
    
    # 6. Decoupled Dataset Splitting & Export (Feedback #2, #3, #7)
    os.makedirs(output_dir, exist_ok=True)
    
    # A. Classification Dataset
    clf_output_path = os.path.join(output_dir, 'hotel_bookings_clf.csv')
    df_final.to_csv(clf_output_path, index=False)
    print(f"Saved classification dataset to `../data/processed/hotel_bookings_clf.csv`")
    
    # B. Regression Dataset (Non-canceled bookings with strict ADR filtering)
    df_reg = df_final[df_final['is_canceled'] == 0].copy()
    df_reg = df_reg[(df_reg['adr'] > 0) & (df_reg['adr'] < 500)]
    
    reg_output_path = os.path.join(output_dir, 'hotel_bookings_reg.csv')
    df_reg.to_csv(reg_output_path, index=False)
    print(f"Saved regression dataset to `../data/processed/hotel_bookings_reg.csv`")
    
    # 7. Final Sanity Check and Verification Audit (Feedback #8)
    print("\n" + "=" * 40)
    print("FINAL PIPELINE VERIFICATION AUDIT")
    print("=" * 40)
    print(f"Raw dataset loaded successfully  : True (Shape: {df_raw.shape})")
    print(f"Classification dataset (CLF) shape: {df_final.shape[0]} rows / {df_final.shape[1]} columns")
    print(f"Regression dataset (REG) shape    : {df_reg.shape[0]} rows / {df_reg.shape[1]} columns")
    print(f"Total Remaining Missing Values   : {df_final.isnull().sum().sum()}")
    print(f"Total Remaining Object Columns   : {len(df_final.select_dtypes(include=['object']).columns)}")
    print("=" * 40)

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    preprocess_hotel_booking_data(
        raw_data_path=os.path.join(project_root, "data", "raw", "hotel_bookings.csv"),
        output_dir=os.path.join(project_root, "data", "processed")
    )
