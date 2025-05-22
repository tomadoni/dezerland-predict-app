import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# === Load Model ===
model = joblib.load("linear_regression_model.pkl")

# === Page Config ===
st.set_page_config(page_title="🚗 Dezerland Park Visitor Predictor", layout="centered")

st.title("🚗 Dezerland Park Visitor Predictor")
st.markdown("Predict daily visitor count using historical trends and weather patterns at the Dezerland Park.")

st.divider()

# === User Inputs ===
col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("📆 Month", range(1, 13), format_func=lambda m: datetime(2024, m, 1).strftime("%B"))
    day_of_week = st.selectbox("🗓️ Day of Week", 
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d])

with col2:
    precipitation = st.slider("🌧️ Precipitation (inches)", min_value=0.0, max_value=3.0, step=0.1, value=0.0)
    school_break = st.radio("🏫 Is School Out?", [0, 1], format_func=lambda x: "Yes" if x else "No")

is_weekend = 1 if day_of_week in [5, 6] else 0

# === Predict ===
input_df = pd.DataFrame([{
    "Month": month,
    "Precipitation": precipitation,
    "SchoolBreak": school_break,
    "DayOfWeek": day_of_week,
    "IsWeekend": is_weekend
}])

predicted_visitors = int(model.predict(input_df)[0])

st.metric(label="🎯 Predicted Visitors", value=f"{predicted_visitors:,}")

# === Show Inputs (optional)
with st.expander("🔍 View Model Input Details"):
    st.dataframe(input_df.T.rename(columns={0: "Input Value"}))

st.divider()
st.markdown("🧠 Powered by a Linear Regression model trained on Dezerland Park visitor data (May 2024–Apr 2025).")

