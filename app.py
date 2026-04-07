import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide", page_icon="📈")

# Clean UI CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    div.stButton > button:first-child {
        background-color: #004aad;
        color: white;
        border-radius: 5px;
    }
    div.stButton > button:first-child:hover {
        background-color: #003380;
    }
    h1, h2, h3 {color: #1f2937;}
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    path = r"c:\Users\nivas\Desktop\Sales forecasting\train_engineered.csv"
    df = pd.read_csv(path, usecols=['Store', 'Date', 'Sales', 'IsStateHoliday', 'IsWeekend'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

with st.spinner("Loading engineered dataset (might take a few seconds)..."):
    try:
        df_full = load_data()
    except FileNotFoundError:
        st.error("Engineered data file not found. Ensure Phase 2 is complete.")
        st.stop()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Configure your forecast projection below.")

# Store Selection
stores = sorted(df_full['Store'].unique())
selected_store = st.sidebar.selectbox("Select Store ID", stores, index=0)

# Horizon Selection
horizon_map = {"30 Days (1 Month)": 30, "60 Days (2 Months)": 60, "90 Days (1 Quarter)": 90}
selected_horizon_label = st.sidebar.radio("Select Forecast Horizon", list(horizon_map.keys()))
horizon = horizon_map[selected_horizon_label]

st.sidebar.markdown("---")
st.sidebar.info("The models utilize Meta Prophet with built-in holiday and weekend cyclic regressors.")

# -------------------------
# Main Body
# -------------------------
st.title(f"📈 Store {selected_store} Sales Forecast")
st.markdown(f"Running an interactive **Meta Prophet** projection targeting the next **{horizon} days** with 80% confidence intervals.")

# Filter Data
store_df = df_full[df_full['Store'] == selected_store].copy()
store_df = store_df.sort_values('Date').dropna(subset=['Sales'])

# Train Prophet Model
with st.spinner("Fitting Prophet architecture..."):
    # Rename for Prophet
    prophet_df = store_df[['Date', 'Sales', 'IsStateHoliday', 'IsWeekend']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    
    # Define Holidays
    holidays_df = pd.DataFrame({
        'holiday': 'state_holiday',
        'ds': store_df[store_df['IsStateHoliday'] == 1]['Date'],
        'lower_window': 0,
        'upper_window': 1,
    })
    
    # Initialize Model
    m = Prophet(holidays=holidays_df, daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays(country_name='DE') # Rossmann is German
    m.add_regressor('IsWeekend')
    
    # Fit Model
    m.fit(prophet_df)
    
    # Predict Future
    future = m.make_future_dataframe(periods=horizon)
    # Populate future regressors (IsWeekend)
    future['IsWeekend'] = future['ds'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    
    forecast = m.predict(future)

# Metrics
st.markdown("### Projected Metrics")
col1, col2, col3 = st.columns(3)

historical_avg = store_df['Sales'].mean()
forecasted_avg = forecast.tail(horizon)['yhat'].mean()
growth = ((forecasted_avg - historical_avg) / historical_avg) * 100

col1.metric("Historical Avg Dailies", f"{historical_avg:,.0f}€")
col2.metric("Forecasted Avg (Next Period)", f"{forecasted_avg:,.0f}€", f"{growth:+.1f}%")
col3.metric("Projected Total Volume", f"{forecast.tail(horizon)['yhat'].sum():,.0f}€")

st.markdown("---")

# Visualizations
st.markdown("### Interactive Projection")
fig = plot_plotly(m, forecast)
fig.update_layout(
    title=None,
    xaxis_title="Date",
    yaxis_title="Sales Volume",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# Raw Data Expander
with st.expander("View Forecast Data Details"):
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).rename(
        columns={'ds': 'Date', 'yhat': 'Forecast', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
    ))

