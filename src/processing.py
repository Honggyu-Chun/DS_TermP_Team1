from pathlib import Path

import pandas as pd


def preprocess_hotel_booking_data(
    raw_data_path: str = "data/raw/hotel_bookings.csv",
    output_dir: str = "data/processed",
) -> dict:
    """
    Run the hotel booking preprocessing steps and save model-ready datasets.
    This keeps the script and notebook preprocessing logic consistent.
    """
    print("=" * 60)
    print("Starting preprocessing...")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    raw_path = Path(raw_data_path)
    if not raw_path.is_absolute() and not raw_path.exists():
        raw_path = project_root / raw_path

    processed_dir = Path(output_dir)
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir
    
    # 1. Raw Data Loading & Validation
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at target path: {raw_path}")
        
    df_raw = pd.read_csv(raw_path)
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    print(f"Loaded raw dataset successfully. Initial Shape: {df.shape}")
    
    # 2. Data cleaning and anomaly handling
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
    # This corresponds to total_stay_days in the project proposal.
    # Keep the existing column name for notebook and report compatibility.
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)
    
    # Consolidate high cardinality country data to prevent sparse matrix explosion
    top_10_countries = df['country'].value_counts().nlargest(10).index
    df['country'] = df['country'].apply(lambda x: x if x in top_10_countries else 'Other')
    
    # 4. Data Leakage Prevention
    # Drop columns that are determined after the booking outcome
    leak_cols = ['reservation_status', 'reservation_status_date']
    df.drop([c for c in leak_cols if c in df.columns], axis=1, inplace=True)
    
    # 5. Categorical one-hot encoding
    cat_cols = [
        'hotel', 'meal', 'market_segment', 'distribution_channel', 
        'deposit_type', 'customer_type', 'arrival_date_month', 'country',
        'reserved_room_type', 'assigned_room_type'
    ]
    df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Convert boolean dummy columns to 0/1 integers.
    bool_cols = df_final.select_dtypes(include=['bool']).columns
    df_final[bool_cols] = df_final[bool_cols].astype(int)
    
    # 6. Save separate classification and regression datasets
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Classification Dataset
    clf_output_path = processed_dir / 'hotel_bookings_clf.csv'
    df_final.to_csv(clf_output_path, index=False)
    print(f"Saved classification dataset to `{clf_output_path}`")
    
    # B. Regression Dataset (Non-canceled bookings with strict ADR filtering)
    df_reg = df_final[df_final['is_canceled'] == 0].copy()
    df_reg = df_reg[(df_reg['adr'] > 0) & (df_reg['adr'] < 500)]
    
    reg_output_path = processed_dir / 'hotel_bookings_reg.csv'
    df_reg.to_csv(reg_output_path, index=False)
    print(f"Saved regression dataset to `{reg_output_path}`")

    missing_values_after_processing = int(df_final.isnull().sum().sum())
    object_columns_after_processing = len(df_final.select_dtypes(include=['object']).columns)
    
    # 7. Final sanity check
    print("\n" + "=" * 40)
    print("Preprocessing check")
    print("=" * 40)
    print(f"Raw dataset loaded successfully  : True (Shape: {df_raw.shape})")
    print(f"Classification dataset (CLF) shape: {df_final.shape[0]} rows / {df_final.shape[1]} columns")
    print(f"Regression dataset (REG) shape    : {df_reg.shape[0]} rows / {df_reg.shape[1]} columns")
    print(f"Total Remaining Missing Values   : {missing_values_after_processing}")
    print(f"Total Remaining Object Columns   : {object_columns_after_processing}")
    print("=" * 40)

    return {
        "classification_path": str(clf_output_path),
        "regression_path": str(reg_output_path),
        "raw_shape": df_raw.shape,
        "classification_shape": df_final.shape,
        "regression_shape": df_reg.shape,
        "missing_values_after_processing": missing_values_after_processing,
        "object_columns_after_processing": object_columns_after_processing,
    }


if __name__ == "__main__":
    result = preprocess_hotel_booking_data()
    print("Classification path:", result["classification_path"])
    print("Regression path:", result["regression_path"])
