import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION (Use Environment Variables for Security) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

def generate_weekly_summary():
    # Load sample data (Store 1)
    df = pd.read_csv("train_engineered.csv", usecols=['Store', 'Date', 'Sales', 'IsStateHoliday', 'IsWeekend'])
    df['Date'] = pd.to_datetime(df['Date'])
    store_1 = df[df['Store'] == 1].tail(365) # Last year of data
    
    # Simple Prophet Forecast
    prophet_df = store_1[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    
    # Generate Chart
    plt.figure(figsize=(10, 6))
    plt.plot(store_1['Date'], store_1['Sales'], label='Actual', color='blue')
    plt.plot(forecast['ds'].tail(7), forecast['yhat'].tail(7), label='Forecast', color='red', linestyle='--')
    plt.title("Weekly Sales Forecast Digest - Store 1")
    plt.legend()
    plt.savefig("digest_chart.png")
    plt.close()
    
    total_forecast = forecast['yhat'].tail(7).sum()
    return total_forecast

def send_email(total_sales):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Skipping email: Credentials not found.")
        return

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = "📈 Weekly Rossmann Sales Digest"

    body = f"""
    <h2>Your Weekly Forecasting Report is Ready</h2>
    <p>The AI model has completed its weekly run. Here is the summary for Store 1:</p>
    <ul>
        <li><b>Forecasted Revenue (Next 7 Days):</b> {total_sales:,.2f}€</li>
        <li><b>Status:</b> Healthy</li>
    </ul>
    <p>Please find the visual projection attached below.</p>
    """
    msg.attach(MIMEText(body, 'html'))

    # Attach Chart
    with open("digest_chart.png", "rb") as f:
        img_data = f.read()
        image = MIMEImage(img_data, name="digest_chart.png")
        msg.attach(image)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    sales = generate_weekly_summary()
    send_email(sales)
