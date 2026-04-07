import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from prophet import Prophet
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
base_dir = r"c:\Users\nivas\Desktop\Sales forecasting"
data_path = os.path.join(base_dir, "train_engineered.csv")

def evaluate(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f"{model_name} Performance:")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}\n")
    return mae, rmse

def train_models():
    print("Loading engineered data...")
    df = pd.read_csv(data_path)
    
    # Selecting Store 1 for performance reasons
    print("Filtering down to Store 1...")
    store_df = df[df['Store'] == 1].copy()
    
    # Define Regressors
    regressors = ['Promo', 'IsStateHoliday', 'IsWeekend', 'Sales_Lag7', 'Sales_Roll7_Mean']
    
    # Process dates and handle NAs
    store_df['Date'] = pd.to_datetime(store_df['Date'])
    store_df = store_df.sort_values('Date')
    store_df.dropna(subset=['Sales', 'Date'] + regressors, inplace=True)
    
    # Train-test split (Let's hold out the last 42 days, ~6 weeks, a common Kaggle holdout)
    test_size = 42
    train = store_df.iloc[:-test_size]
    test = store_df.iloc[-test_size:]
    
    print(f"Training set: {train['Date'].min().date()} to {train['Date'].max().date()} ({len(train)} days)")
    print(f"Test set: {test['Date'].min().date()} to {test['Date'].max().date()} ({len(test)} days)")
    
    # -----------------------
    # 1. SARIMA (auto_arima)
    # -----------------------
    print("\nTraining SARIMA with auto_arima...")
    X_train_arima = train[regressors]
    y_train_arima = train['Sales']
    
    X_test_arima = test[regressors]
    y_test_actual = test['Sales']
    
    # Note: We use m=7 for weekly seasonality
    arima_model = auto_arima(
        y=y_train_arima,
        X=X_train_arima,
        seasonal=True,
        m=7, # weekly seasonality
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=True
    )
    
    print(arima_model.summary())
    
    print("Forecasting with SARIMA...")
    arima_pred = arima_model.predict(n_periods=test_size, X=X_test_arima)
    
    # -----------------------
    # 2. PROPHET
    # -----------------------
    print("\nTraining Prophet...")
    # Prophet requires columns 'ds' and 'y'
    prophet_train = train[['Date', 'Sales'] + regressors].rename(columns={'Date': 'ds', 'Sales': 'y'})
    prophet_test = test[['Date', 'Sales'] + regressors].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Handle Holidays DataFrame for Prophet
    holidays_df = pd.DataFrame({
        'holiday': 'state_holiday',
        'ds': store_df[store_df['IsStateHoliday'] == 1]['Date'],
        'lower_window': 0,
        'upper_window': 1,
    })
    
    m = Prophet(holidays=holidays_df)
    
    # Add Regressors
    for reg in regressors:
        if reg != 'IsStateHoliday':  # we can exclude it from regressors if handled in holidays, but no harm leaving it or dropping it. We'll drop to avoid collinearity.
            m.add_regressor(reg)
            
    m.fit(prophet_train)
    
    print("Forecasting with Prophet...")
    # Prepare future dataframe and add regressor columns to it
    future = prophet_test.drop('y', axis=1)
    prophet_forecast = m.predict(future)
    prophet_pred = prophet_forecast['yhat']
    
    # -----------------------
    # EVALUATION
    # -----------------------
    print("\n--- Evaluation on Test Window ---")
    evaluate(y_test_actual, arima_pred, "SARIMA")
    evaluate(y_test_actual, prophet_pred, "Prophet")
    
    # PLOT
    plt.figure(figsize=(14, 7))
    plt.plot(test['Date'], y_test_actual, label='Actual Sales', color='black', linewidth=2)
    plt.plot(test['Date'], arima_pred, label='SARIMA Forecast', linestyle='dashed', color='blue')
    plt.plot(test['Date'], prophet_pred.values, label='Prophet Forecast', linestyle='dashdot', color='red')
    plt.title('SARIMA vs Prophet: Sales Forecast against Actuals (Last 42 Days)')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_path = os.path.join(base_dir, "model_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved forecast comparison plot to {plot_path}")

if __name__ == "__main__":
    train_models()
