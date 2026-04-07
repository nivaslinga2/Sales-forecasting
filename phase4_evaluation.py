import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import os
import warnings
warnings.filterwarnings('ignore')

base_dir = r"c:\Users\nivas\Desktop\Sales forecasting"
data_path = os.path.join(base_dir, "train_engineered.csv")

def evaluate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    actual_vals = np.array(actual)
    pred_vals = np.array(predicted)
    
    # Handle zeros in actuals for MAPE by filtering them out, to avoid astronomical values.
    mask = actual_vals != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(actual_vals[mask], pred_vals[mask]) * 100
    else:
        mape = np.nan
        
    return mae, rmse, mape

def phase4():
    print("Loading engineered data...")
    df = pd.read_csv(data_path)
    store_df = df[df['Store'] == 1].copy()
    
    regressors = ['Promo', 'IsStateHoliday', 'IsWeekend', 'Sales_Lag7', 'Sales_Roll7_Mean']
    store_df['Date'] = pd.to_datetime(store_df['Date'])
    store_df = store_df.sort_values('Date')
    store_df.dropna(subset=['Sales', 'Date'] + regressors, inplace=True)
    
    test_size = 42
    train = store_df.iloc[:-test_size]
    test = store_df.iloc[-test_size:]
    
    y_test = test['Sales']
    
    print("\nCalculating Naive Baseline (Last week's sales)...")
    naive_pred = test['Sales_Lag7']
    naive_metrics = evaluate_metrics(y_test, naive_pred)
    
    print("\nTraining SARIMA(2,0,1)(2,0,2)[7] (Optimal parameters from Phase 3)...")
    sarima_model = SARIMAX(
        endog=train['Sales'].values,
        exog=train[regressors].values,
        order=(2, 0, 1),
        seasonal_order=(2, 0, 2, 7)
    )
    sarima_fit = sarima_model.fit(disp=False)
    sarima_pred = sarima_fit.predict(start=len(train), end=len(train)+len(test)-1, exog=test[regressors].values)
    sarima_metrics = evaluate_metrics(y_test.values, sarima_pred)
    
    print("\nTraining Prophet...")
    prophet_train = train[['Date', 'Sales'] + regressors].rename(columns={'Date': 'ds', 'Sales': 'y'})
    prophet_test = test[['Date', 'Sales'] + regressors].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    holidays_df = pd.DataFrame({
        'holiday': 'state_holiday',
        'ds': store_df[store_df['IsStateHoliday'] == 1]['Date'],
        'lower_window': 0,
        'upper_window': 1,
    })
    
    m = Prophet(holidays=holidays_df)
    for reg in regressors:
        m.add_regressor(reg)
    m.fit(prophet_train)
    
    future = prophet_test.drop('y', axis=1)
    prophet_forecast = m.predict(future)
    prophet_pred = prophet_forecast['yhat']
    prophet_metrics = evaluate_metrics(y_test, prophet_pred)
    
    # Print Markdown Table
    print("\n" + "="*60)
    print("## Phase 4: Model Comparison Table")
    print("| Model | MAE | RMSE | MAPE (%) |")
    print("|---|---|---|---|")
    print(f"| Baseline (Naive Lag7) | {naive_metrics[0]:.2f} | {naive_metrics[1]:.2f} | {naive_metrics[2]:.2f}% |")
    print(f"| SARIMA (2,0,1)x(2,0,2)[7] | {sarima_metrics[0]:.2f} | {sarima_metrics[1]:.2f} | {sarima_metrics[2]:.2f}% |")
    print(f"| Meta Prophet | {prophet_metrics[0]:.2f} | {prophet_metrics[1]:.2f} | {prophet_metrics[2]:.2f}% |")
    print("="*60)

if __name__ == "__main__":
    phase4()
