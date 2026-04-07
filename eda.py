import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import os
import warnings
warnings.filterwarnings("ignore")

# Set paths
base_dir = r"c:\Users\nivas\Desktop\Sales forecasting"
train_path = os.path.join(base_dir, "train.csv")

def main():
    print("Loading data...")
    df = pd.read_csv(train_path, low_memory=False)
    
    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Plot sales over time (Aggregated daily sales)
    print("Aggregating sales over time...")
    daily_sales = df.groupby('Date')['Sales'].sum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sales, label='Total Daily Sales')
    plt.title('Total Sales over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "sales_over_time.png"))
    plt.close()
    print("Saved 'sales_over_time.png'")
    
    # 2. Identify stores with most variance
    print("\nCalculating store variance...")
    store_variance = df.groupby('Store')['Sales'].var().sort_values(ascending=False)
    print("\nTop 5 Stores with highest Sales Variance:")
    print(store_variance.head())
    
    # 3. Check for missing dates
    print("\nChecking for missing dates...")
    full_date_range = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max())
    missing_dates = full_date_range.difference(daily_sales.index)
    if len(missing_dates) == 0:
        print("No missing dates found in the overall timeline!")
    else:
        print(f"Found {len(missing_dates)} missing dates. Here are the first few:")
        print(missing_dates[:5])
        
    # Fill any missing aggregated dates with 0 or interpolate before decomposition if needed
    daily_sales = daily_sales.asfreq('D')
    if daily_sales.isnull().sum() > 0:
        daily_sales = daily_sales.interpolate(method='linear')
    
    # 4. Decompose the time series
    print("\nDecomposing the time series...")
    # Period 7 for weekly seasonality or 365 for yearly
    # We will use freq=7 to capture weekly trends which are very prominent in retail
    decompose_result = seasonal_decompose(daily_sales, model='additive', period=7)
    
    fig = decompose_result.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    plt.savefig(os.path.join(base_dir, "seasonal_decompose.png"))
    plt.close()
    print("Saved 'seasonal_decompose.png'")
    
    # 5. Run ADF test to check stationarity
    print("\nRunning Augmented Dickey-Fuller (ADF) Test...")
    adf_result = adfuller(daily_sales)
    
    print("ADF Statistic:", adf_result[0])
    print("p-value:", adf_result[1])
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value}")
        
    if adf_result[1] < 0.05:
        print("=> Reject the null hypothesis: The time series is stationary.")
    else:
        print("=> Fail to reject the null hypothesis: The time series is non-stationary.")

if __name__ == "__main__":
    main()
