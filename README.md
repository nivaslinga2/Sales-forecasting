# 📈 Rossmann Store Sales Forecasting

> **End-to-end time-series forecasting pipeline** — from raw Kaggle data to a deployed FastAPI microservice with automated weekly retraining via GitHub Actions.

---

## 🎯 What This Project Does

Predicts daily sales for **1,115 Rossmann drug stores** across Germany using historical transactional data (2013–2015). The pipeline covers the full ML lifecycle: **EDA → Feature Engineering → Model Training → Evaluation → Dashboard → Deployment**.

---

## 🧠 Interview Quick Reference

### The Problem
Rossmann operates 3,000+ drug stores across Europe. Store managers need accurate **6-week sales forecasts** to plan staffing, inventory, and promotions. The challenge: sales are influenced by promotions, holidays, seasonality, competition, and store-specific behavior.

### The Data (Kaggle)
| File | Rows | Description |
|---|---|---|
| `train.csv` | 1,017,209 | Historical daily sales per store (Jan 2013 – Jul 2015) |
| `store.csv` | 1,115 | Store metadata (type, assortment, competition distance) |
| `test.csv` | 41,088 | Stores × dates to predict (Aug–Sep 2015) |

### Key Columns
- **Sales**: Target variable (daily revenue in €)
- **Customers**: Daily foot traffic
- **Open/Promo/StateHoliday/SchoolHoliday**: Binary flags
- **StoreType** (a–d) / **Assortment** (a–c): Store categories
- **CompetitionDistance**: Meters to nearest competitor

---

## 🔬 Phase-by-Phase Breakdown

### Phase 1 — Exploratory Data Analysis (`eda.py`)
| What I Did | Why It Matters |
|---|---|
| Plotted total daily sales over time | Reveals overall trend + December spikes |
| Identified top-variance stores (817, 1114, 251) | High-variance stores need special handling |
| Checked for missing dates | Confirmed 0 gaps — clean continuous timeline |
| Seasonal decomposition (period=7) | Proved strong **weekly seasonality** in retail |
| **ADF Test**: statistic = -4.76, p-value = 0.00006 | **Series is stationary** → no differencing needed (d=0) |

**Interview talking point:** *"I ran an Augmented Dickey-Fuller test and confirmed the aggregated series is stationary with a p-value of 6.4e-5, which informed my ARIMA parameter selection — specifically that d=0."*

---

### Phase 2 — Feature Engineering (`feature_engineering.py`)
| Feature | Type | Rationale |
|---|---|---|
| `Sales_Lag7`, `Sales_Lag14` | Lag | Last week/fortnight sales as direct signals |
| `Sales_Roll7_Mean`, `Sales_Roll7_Std` | Rolling window | Captures recent momentum per store |
| `Year`, `Month`, `Day`, `WeekOfYear` | Calendar | Encodes temporal patterns |
| `IsWeekend`, `IsStateHoliday` | Binary flag | Captures behavioral shifts |
| `Promo` | Binary flag | Promotion impact |
| Store metadata merge | Contextual | CompetitionDistance, StoreType, Assortment |

**Key design decision:** Rolling features use `shift(1)` before the window to **prevent data leakage** — today's sales never appear in today's features.

**Interview talking point:** *"I was careful about data leakage. My rolling mean shifts by one day before computing the 7-day window, so the model never sees the current day's target in its input features."*

---

### Phase 3 — Model Training (`model_training.py`)
#### SARIMA (Statistical)
- Used `pmdarima.auto_arima` with `m=7` (weekly seasonality)
- Best fit: **SARIMA(2,0,1)(2,0,2)[7]** with exogenous regressors
- Captures linear autoregressive patterns

#### Meta Prophet (Additive)
- Built-in weekly + yearly seasonality
- German national holidays via `add_country_holidays('DE')`
- Added `IsWeekend`, `Promo`, `Sales_Lag7`, `Sales_Roll7_Mean` as regressors
- Handles missing data and outliers gracefully

**Interview talking point:** *"I chose Prophet as the primary model because it handles multiple seasonalities natively and allows me to inject domain knowledge through holiday calendars and custom regressors — which is exactly what retail forecasting needs."*

---

### Phase 4 — Model Comparison (`phase4_evaluation.py`)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| **Baseline (Naive Lag7)** | 981.45 | 1162.32 | 26.29% |
| **SARIMA (2,0,1)×(2,0,2)[7]** | 491.46 | 594.05 | 11.71% |
| **Meta Prophet** | **305.61** | **381.16** | **7.03%** |

**Why the baseline matters:** *"I always include a naive baseline because a model is only valuable if it beats simple heuristics. My Prophet model cut the error by 73% compared to just repeating last week's sales."*

**Interview talking point:** *"Prophet achieved a 7% MAPE on the holdout set — meaning on average, my predictions are within 7% of actual sales. That's stro### Phase 5 — Advanced Dashboard & Analytics (`app.py`)
Interactive web app upgraded with **Premium Business Intelligence** features:
- **KPI Summary Header**: Total Forecast Revenue, 92.9% Model Accuracy (MAPE-based), Top Historical Store, and WoW (Week-over-Week) Growth trends.
- **Data Quality Scorecard**: Real-time audit showing Data Completeness %, Anomaly Detection (3-Sigma outliers), and Historical Range.
- **Model Explainability**: "Why this forecast?" section using Prophet component decomposition to show the relative impact of Trend, Seasonality, and Holidays.
- **Interactive Projections**: Plotly chart with **80% and 95% dual confidence bands** for risk assessment.
- **One-Click Exports**: Download precise forecast tables as CSV for spreadsheet integration.
- **Branded Executive Reports**: sidebar button to generate a professional **3-page PDF report** with automated branding, benchmarks, and projections.

```bash
streamlit run app.py
```

---

### Phase 6 — Deployment & Automation (`api.py`, `email_digest.py`, GitHub Actions)

#### FastAPI Microservice
Standardized REST API for real-time predictions.

#### Automated Weekly Email Digest (`email_digest.py`)
- **Self-Service Automation**: Script that runs in the cloud every Monday.
- **Email Delivery**: Sends an HTML summary with the latest forecast chart and revenue totals to stakeholders via `smtplib`.

#### CI/CD & Cron Workflows
- **`ci.yml`**: Build and Test pipeline.
- **`retrain.yml`**: Weekly model retraining.
- **`weekly_digest.yml`**: **NEW** Automated email report trigger via GitHub Actions cron.

---

## 📂 Project Structure
```
Sales forecasting/
├── eda.py                    # Phase 1: EDA + ADF test
├── feature_engineering.py    # Phase 2: Lag, rolling, calendar features
├── model_training.py         # Phase 3: SARIMA + Prophet training
├── phase4_evaluation.py      # Phase 4: Baseline vs model comparison
├── app.py                    # Phase 5: Advanced BI Dashboard
├── report_generator.py       # NEW: PDF Executive Report Engine
├── email_digest.py           # NEW: Automated Email Summary Script
├── api.py                    # Phase 6: FastAPI prediction endpoint
├── retrain.py                # Automated retraining script
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── .github/
│   └── workflows/
│       ├── ci.yml            # CI pipeline
│       ├── retrain.yml       # Weekly cron retraining
│       └── weekly_digest.yml # Weekly email automation
├── train.csv                 # Raw training data (1M+ rows)
├── store.csv                 # Store metadata
├── test.csv                  # Kaggle test set
└── train_engineered.csv      # Feature-engineered dataset
```

---

## 🚀 Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run EDA
python eda.py

# 3. Engineer features
python feature_engineering.py

# 4. Train & evaluate models
python phase4_evaluation.py

# 5. Launch dashboard
streamlit run app.py

# 6. Start API
uvicorn api:app --reload
```

---

## 🛠 Tech Stack
| Category | Tools |
|---|---|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Statistical Models** | statsmodels (SARIMAX), pmdarima (auto_arima) |
| **ML Models** | Meta Prophet |
| **Stationarity Test** | ADF (Augmented Dickey-Fuller) |
| **Dashboard** | Streamlit |
| **API** | FastAPI + Uvicorn |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |

---

## 💡 Key Interview Questions & Answers

**Q: Why Prophet over ARIMA?**
> Prophet handles multiple seasonalities (weekly + yearly) natively, lets me inject holiday calendars, and is robust to missing data. ARIMA required manual parameter tuning and couldn't capture complex seasonal interactions as cleanly.

**Q: How did you prevent data leakage?**
> All lag and rolling features use `shift(1)` before computing windows. The train/test split is strictly temporal — no future data ever leaks into training.

**Q: How do you know your model is actually good?**
> I benchmarked against a naive baseline (repeat last week's sales). The baseline had 26% MAPE; Prophet achieved 7% — a 73% relative improvement.

**Q: How would you scale this to all 1,115 stores?**
> Two approaches: (1) Train individual Prophet models per store in parallel using multiprocessing, or (2) Use a global gradient boosting model (XGBoost/LightGBM) with store ID as a feature for cross-store learning.

**Q: What would you improve with more time?**
> Add XGBoost ensemble, implement proper cross-validation with TimeSeriesSplit, build a model registry for versioning, and add monitoring dashboards for prediction drift detection.
