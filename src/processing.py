import pandas as pd
import numpy as np
import os

def preprocess_hotel_data(input_path):
    """
    Main preprocessing function that cleans data and performs feature engineering.
    Reflects Team Lead's feedback on Agent/Company binary flags and leakage prevention.
    """
    # Load raw data 
    df = pd.read_csv(input_path)
    
    # 1. Handle Missing Values & Binary Indicators
    # Convert IDs to binary existence flags to prevent model numeric bias
    df['has_agent'] = df['agent'].notnull().astype(int)
    df['has_company'] = df['company'].notnull().astype(int)
    df.drop(['agent', 'company'], axis=1, inplace=True)
    
    df['children'] = df['children'].fillna(0)
    df['country'] = df['country'].fillna('Unknown')

    # 2. Feature Engineering [cite: 253, 254]
    df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)

    # 3. Prevent Data Leakage
    # Dropping columns determined after the outcome is known
    df.drop(['reservation_status', 'reservation_status_date'], axis=1, inplace=True)

    # 4. Categorical Encoding
    # Include all room type columns as per feedback
    cat_cols = [
        'hotel', 'meal', 'market_segment', 'distribution_channel', 
        'deposit_type', 'customer_type', 'arrival_date_month', 'country',
        'reserved_room_type', 'assigned_room_type'
    ]
    df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df_final

def save_processed_datasets(df_final, output_dir='./data/processed/'):
    """
    Splits the processed data into CLF and REG datasets and saves them. 
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Classification Dataset
    df_final.to_csv(os.path.join(output_dir, 'hotel_bookings_clf.csv'), index=False)

    # Regression Dataset with strict ADR filtering for actual guests
    df_reg = df_final[df_final['is_canceled'] == 0].copy()
    df_reg = df_reg[(df_reg['adr'] > 0) & (df_reg['adr'] < 500)] # Target stability
    df_reg.to_csv(os.path.join(output_dir, 'hotel_bookings_reg.csv'), index=False)

if __name__ == "__main__":
    
    INPUT = './data/raw/hotel_bookings.csv' 

    final_df = preprocess_hotel_data(INPUT)
    
    save_processed_datasets(final_df, output_dir='./data/processed/')
    
    print("Preprocessing Script Execution Complete.")