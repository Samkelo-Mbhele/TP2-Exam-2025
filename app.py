import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from prophet import Prophet

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(page_title="Crime Data Forecast Dashboard", layout="wide")

st.title("🚔 Crime Data Forecasting Dashboard")

# ---------------------------------------------
# AUTO-LOAD SAMPLE DATA
# ---------------------------------------------
# (Replace with your real CSV path if needed)
data = {
    'date': pd.date_range(start='2020-01-01', periods=36, freq='M'),
    'location': ['Durban', 'Johannesburg', 'Cape Town'] * 12,
    'crime_type': ['Theft', 'Assault', 'Burglary'] * 12,
    'crime_count': [120, 95, 80, 130, 100, 90, 140, 105, 100,
                    160, 120, 110, 170, 125, 115, 180, 130, 120,
                    190, 140, 130, 200, 145, 135, 210, 150, 140,
                    220, 155, 145, 230, 160, 150, 240, 165, 155]
}
crime_df = pd.DataFrame(data)

# ---------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------
locations = st.sidebar.multiselect(
    "Select Location(s)",
    options=crime_df['location'].unique(),
    default=crime_df['location'].unique()
)

crime_types = st.sidebar.multiselect(
    "Select Crime Type(s)",
    options=crime_df['crime_type'].unique(),
    default=crime_df['crime_type'].unique()
)

filtered_df = crime_df[
    (crime_df['location'].isin(locations)) &
    (crime_df['crime_type'].isin(crime_types))
]

# ---------------------------------------------
# DASHBOARD DISPLAY
# ---------------------------------------------
st.subheader("📊 Filtered Crime Data")
st.dataframe(filtered_df, use_container_width=True)

# ---------------------------------------------
# CRIME TYPE DISTRIBUTION BAR CHART
# ---------------------------------------------
st.subheader("🔍 Crime Type Distribution")

crime_counts = (
    filtered_df.groupby('crime_type')['crime_count']
    .sum()
    .reset_index()
)

fig_bar = px.bar(
    crime_counts,
    x='crime_type',
    y='crime_count',
    title="Crime Type Distribution by Count",
    text='crime_count',
    color='crime_type'
)
st.plotly_chart(fig_bar, use_container_width=True)

# ---------------------------------------------
# TIME SERIES FORECASTING USING PROPHET
# ---------------------------------------------
st.subheader("📈 Forecasting Future Crime Trends (Prophet)")

df_forecast = (
    filtered_df.groupby('date')['crime_count']
    .sum()
    .reset_index()
    .rename(columns={'date': 'ds', 'crime_count': 'y'})
)

# Prophet model
model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# Plot forecast
fig_forecast = model.plot(forecast)
st.pyplot(fig_forecast)

# ---------------------------------------------
# INSIGHTS & INTERPRETATION
# ---------------------------------------------
st.subheader("🧠 Insights & Interpretation")

st.markdown("""
### 🔹 How Prophet Solves the Problem
Prophet helps us **forecast future crime trends** by analyzing past data patterns.  
It captures both **seasonal** (e.g., crimes that rise in December) and **trend** (overall increase or decrease) components.

This allows:
- Better **planning of police resources**.
- **Prediction of high-risk months**.
- Data-driven decision making.

### 🔹 Why We Used Prophet
- Handles **missing data** and **irregular time intervals** easily.  
- Automatically models **seasonality and trends**.  
- Great for **crime prediction dashboards** that update monthly.

### 🔹 Why Mention ARIMA
ARIMA is another forecasting model, good for smaller or more consistent datasets.  
However, Prophet is more flexible and interpretable — perfect for public safety analytics.

### 🔹 Datasets Without South Africa
Even when the dataset doesn’t contain South African data,  
Prophet still learns from **global crime behavior patterns** — which helps us  
build and test forecasting logic applicable to South Africa later.

In other words, the **model’s structure and trend detection** remain the same  
— only the **input data changes**, meaning the model is easily adaptable.
""")
