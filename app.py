import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------
# 🎯 Load your trained model
# ------------------------------
model = joblib.load("tcs_stock_model.pkl")  # make sure this file exists in same folder

st.set_page_config(page_title="TCS Stock Price Predictor", page_icon="📈", layout="centered")

st.title("📊 TCS Stock Price Prediction App")
st.markdown("Enter details below to predict the **TCS Closing Price** using your trained Random Forest model.")

# ------------------------------
# 🧮 Input fields
# ------------------------------
st.header("Enter Stock Details")

col1, col2 = st.columns(2)

with col1:
    open_price = st.number_input("Open Price (₹)", min_value=0.0, value=28.0)
    high_price = st.number_input("High Price (₹)", min_value=0.0, value=28.0)
    low_price = st.number_input("Low Price (₹)", min_value=0.0, value=21.0)
    volume = st.number_input("Volume", min_value=0.0, value=200000.0)
    dividends = st.number_input("Dividends", min_value=0.0, value=0.0)

with col2:
    stock_splits = st.number_input("Stock Splits", min_value=0.0, value=0.0)
    year = st.number_input("Year", min_value=2000, max_value=2100, value=datetime.now().year)
    month = st.number_input("Month", min_value=1, max_value=12, value=datetime.now().month)
    day = st.number_input("Day", min_value=1, max_value=31, value=datetime.now().day)
    day_of_week = st.number_input("Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=datetime.now().weekday())

# Auto-calculate date-based features
date = datetime(int(year), int(month), int(day))
day_of_year = date.timetuple().tm_yday
is_month_start = 1 if date.day == 1 else 0
is_month_end = 1 if date.day == pd.Period(date.strftime('%Y-%m')).days_in_month else 0

# ------------------------------
# 🧾 Prepare input for prediction
# ------------------------------
new_prediction = pd.DataFrame({
    "Open": [open_price],
    "High": [high_price],
    "Low": [low_price],
    "Volume": [volume],
    "Dividends": [dividends],
    "Stock Splits": [stock_splits],
    "Year": [year],
    "Month": [month],
    "Day": [day],
    "DayOfWeek": [day_of_week],
    "DayOfYear": [day_of_year],
    "IsMonthStart": [is_month_start],
    "IsMonthEnd": [is_month_end]
})

# ------------------------------
# 🔮 Prediction Button
# ------------------------------
if st.button("Predict Closing Price"):
    predicted_close = model.predict(new_prediction)[0]
    st.success(f"📈 Predicted TCS Closing Price: ₹{predicted_close:.2f}")
    st.caption("Model: Random Forest Regressor")

# ------------------------------
# 🎨 Footer
# ------------------------------
st.markdown("---")
st.markdown("Created by **Satyajit Tekawade** 🧠")
