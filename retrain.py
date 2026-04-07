"""
retrain.py — Automated weekly retraining script.
Triggered by GitHub Actions cron or manually.
Reloads raw data, re-engineers features, and validates the model.
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os, json, datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def retrain():
    print(f"[{datetime.datetime.now()}] Starting scheduled retraining...")

    # Step 1: Load raw data
    train_path = os.path.join(BASE_DIR, "train.csv")
    store_path = os.path.join(BASE_DIR, "store.csv")

    df = pd.read_csv(train_path, low_memory=False)
    store = pd.read_csv(store_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date'])

    # Step 2: Re-engineer features (lightweight version for retraining)
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    df['IsStateHoliday'] = df['StateHoliday'].apply(lambda x: 1 if x in ['a','b','c'] else 0)
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)

    # Step 3: Validate on Store 1 (benchmark store)
    store_df = df[df['Store'] == 1].copy()
    store_df = store_df.sort_values('Date').dropna(subset=['Sales'])

    test_size = 42
    train_data = store_df.iloc[:-test_size]
    test_data = store_df.iloc[-test_size:]

    prophet_df = train_data[['Date', 'Sales', 'IsWeekend']].rename(columns={'Date': 'ds', 'Sales': 'y'})

    m = Prophet(daily_seasonality=False)
    m.add_country_holidays(country_name='DE')
    m.add_regressor('IsWeekend')
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=test_size)
    future['IsWeekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    forecast = m.predict(future)

    y_test = test_data['Sales'].values
    y_pred = forecast.tail(test_size)['yhat'].values

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Step 4: Log results
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "store_benchmark": 1,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "status": "success"
    }

    log_path = os.path.join(BASE_DIR, "retrain_log.json")
    logs = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = json.load(f)
    logs.append(result)
    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    print(f"Retraining complete. MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    print(f"Results logged to {log_path}")

if __name__ == "__main__":
    retrain()
