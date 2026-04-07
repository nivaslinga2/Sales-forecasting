from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from prophet import Prophet
import os

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="An API serving our best Meta Prophet model to forecast store sales.",
    version="1.0.0"
)

# Load data in memory (For production, you'd load pre-trained serialized model artifacts)
DATA_PATH = "train_engineered.csv"

class ForecastRequest(BaseModel):
    store_id: int
    horizon: int = 30

class ForecastResponse(BaseModel):
    store_id: int
    date: str
    forecasted_sales: float
    confidence_lower: float
    confidence_upper: float

@app.on_event("startup")
def load_historical_data():
    global df_full
    if os.path.exists(DATA_PATH):
        df_full = pd.read_csv(DATA_PATH, usecols=['Store', 'Date', 'Sales', 'IsStateHoliday', 'IsWeekend'])
        df_full['Date'] = pd.to_datetime(df_full['Date'])
    else:
        df_full = None

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Forecasting API is online."}

@app.post("/predict", response_model=list[ForecastResponse])
def predict_sales(request: ForecastRequest):
    if df_full is None:
        raise HTTPException(status_code=500, detail="Historical data not found. Cannot fit model.")
        
    store_df = df_full[df_full['Store'] == request.store_id].copy()
    
    if store_df.empty:
        raise HTTPException(status_code=404, detail=f"Store {request.store_id} not found.")

    store_df = store_df.sort_values('Date').dropna(subset=['Sales'])
    
    # Architecture
    prophet_df = store_df[['Date', 'Sales', 'IsStateHoliday', 'IsWeekend']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    holidays_df = pd.DataFrame({
        'holiday': 'state_holiday',
        'ds': store_df[store_df['IsStateHoliday'] == 1]['Date'],
        'lower_window': 0, 'upper_window': 1,
    })
    
    # Train
    m = Prophet(holidays=holidays_df, daily_seasonality=False)
    m.add_country_holidays(country_name='DE')
    m.add_regressor('IsWeekend')
    m.fit(prophet_df)
    
    # Predict
    future = m.make_future_dataframe(periods=request.horizon)
    future['IsWeekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    forecast = m.predict(future)
    
    # Formatting response
    results = forecast.tail(request.horizon)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    response = []
    for _, row in results.iterrows():
        response.append(
            ForecastResponse(
                store_id=request.store_id,
                date=row['ds'].strftime("%Y-%m-%d"),
                forecasted_sales=round(row['yhat'], 2),
                confidence_lower=round(row['yhat_lower'], 2),
                confidence_upper=round(row['yhat_upper'], 2)
            )
        )
    return response

# Run locally via: uvicorn api:app --reload
