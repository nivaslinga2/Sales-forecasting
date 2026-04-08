import pandas as pd
import numpy as np
import os

# Set paths
base_dir = r"c:\Users\nivas\Desktop\Sales forecasting"
train_path = os.path.join(base_dir, "train.csv")
store_path = os.path.join(base_dir, "store.csv")

def feature_engineering():
    print("Loading datasets...")
    df = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    
    # To keep memory footprint low and avoid doing full dataset if requested,
    # we'll do operations on the full dataset here, but it might take a few seconds
    print("Sorting data by Store and Date...")
    df = df.sort_values(['Store', 'Date'])
    
    # 1. Day-of-week/month/year encodings
    print("Creating date features...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # 'DayOfWeek' is already in the data (1-7), but let's make sure
    # IsWeekend flag based on DayOfWeek
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)
    
    # 2. Is-holiday flags (StateHoliday might be string or mixed, need to map to 1/0)
    print("Creating holiday and promotion flags...")
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['IsStateHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x in ['a', 'b', 'c'] else 0)
    
    # Is_SchoolHoliday is already 'SchoolHoliday' (0/1), and Promotion is 'Promo' (0/1)
    
    # Merge Store data (includes StoreType, Assortment, Competition details)
    # We will merge this so that store level features are available too
    df = pd.merge(df, store, on='Store', how='left')
    
    print("Merging Store data and calculating competition distance...")
    # Fill NAs in CompetitionDistance with median
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    
    # 3. Lag Features (grouped by store)
    print("Creating lag features (7 and 14 days)...")
    # Shift sales by 7 and 14 days directly after grouping by Store
    df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
    df['Sales_Lag14'] = df.groupby('Store')['Sales'].shift(14)
    
    # 4. Rolling window statistics (7-day mean/std)
    print("Creating rolling window features (7-day mean, 7-day std)...")
    # Using window=7 (closed='left' or shifting so we don't leak 'today\'s' sales into today's rolling features)
    df['Sales_Roll7_Mean'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )
    df['Sales_Roll7_Std'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).std()
    )
    
    # Fill NAs from lags and rolling with 0
    df.fillna({'Sales_Lag7': 0, 'Sales_Lag14': 0, 'Sales_Roll7_Mean': 0, 'Sales_Roll7_Std': 0}, inplace=True)
    
    print("Feature Engineering complete!")
    print("\nSample of engineered dataset (Columns):")
    print(df.columns.tolist())
    print("\nData shape:", df.shape)
    
    # Show a few rows for one store to see lagged data working correctly
    sample = df[df['Store'] == 1][['Date', 'Sales', 'Sales_Lag7', 'Sales_Roll7_Mean', 'Sales_Roll7_Std', 'IsStateHoliday', 'IsWeekend']].tail(10)
    print("\nPreview of engineered features for Store 1:")
    print(sample)
    
    # Save the processed dataset
    out_path = os.path.join(base_dir, "train_engineered.csv")
    print(f"\nSaving processed data to {out_path}...")
    df.to_csv(out_path, index=False)
    print("Saved successfully!")

if __name__ == "__main__":
    feature_engineering()
