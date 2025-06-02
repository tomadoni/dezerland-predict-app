import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# === Set page config ===
st.set_page_config(page_title="🚗 Dezerland Visitor Predictor", layout="centered")

# === Load Visitor Model ===
visitor_model = joblib.load("linear_regression_model.pkl")

# === UI Layout ===
st.title("🚗 Dezerland Park Visitor Predictor")
st.markdown("Predict daily visitor count using weather and calendar inputs.")

col1, col2 = st.columns(2)

with col1:
    month = st.selectbox("📆 Month", range(1, 13),
                         format_func=lambda m: datetime(2024, m, 1).strftime("%B"),
                         key="vis_month")
    day_of_week = st.selectbox("🗓️ Day of Week", list(range(7)),
                               format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
                               key="vis_day")

with col2:
    precipitation = st.slider("🌧️ Precipitation (inches)", 0.0, 3.0, 0.0, 0.1, key="vis_precip")
    school_break = st.radio("🏫 Is School Out?", [0, 1],
                            format_func=lambda x: "Yes" if x else "No", key="vis_school")

is_weekend = 1 if day_of_week in [5, 6] else 0

# === Input DataFrame for Prediction ===
visitor_input = pd.DataFrame([{
    "Month": month,
    "Precipitation": precipitation,
    "SchoolBreak": school_break,
    "DayOfWeek": day_of_week,
    "IsWeekend": is_weekend
}])

# === Make Prediction ===
predicted_visitors = int(visitor_model.predict(visitor_input)[0])
st.metric("🎯 Predicted Visitors", f"{predicted_visitors:,}")

with st.expander("🔍 View Input Data"):
    st.dataframe(visitor_input.T.rename(columns={0: "Value"}))

