
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# === Set page config (only once) ===
st.set_page_config(page_title="🎡 Dezerland Predictor Dashboard", layout="centered")

# === Load Models ===
visitor_model = joblib.load("linear_regression_model.pkl")
revenue_model_bundle = joblib.load("revenue_per_visitor_model.pkl")
revenue_model = revenue_model_bundle["model"]
revenue_features = revenue_model_bundle["features"]

# === Layout ===
tab1, tab2 = st.tabs(["👥 Visitor Predictor", "💵 OEP Revenue Predictor"])

# === Tab 1: Visitor Predictor ===
with tab1:
    st.title("🚗 Dezerland Park Visitor Predictor")
    st.markdown("Predict daily visitor count using weather and calendar inputs.")

    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("📆 Month", range(1, 13), format_func=lambda m: datetime(2024, m, 1).strftime("%B"), key="vis_month")
        day_of_week = st.selectbox("🗓️ Day of Week", list(range(7)), format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d], key="vis_day")

    with col2:
        precipitation = st.slider("🌧️ Precipitation (inches)", 0.0, 3.0, 0.0, 0.1, key="vis_precip")
        school_break = st.radio("🏫 Is School Out?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="vis_school")

    is_weekend = 1 if day_of_week in [5, 6] else 0

    visitor_input = pd.DataFrame([{
        "Month": month,
        "Precipitation": precipitation,
        "SchoolBreak": school_break,
        "DayOfWeek": day_of_week,
        "IsWeekend": is_weekend
    }])

    predicted_visitors = int(visitor_model.predict(visitor_input)[0])
    st.metric("🎯 Predicted Visitors", f"{predicted_visitors:,}")

    with st.expander("🔍 View Input Data"):
        st.dataframe(visitor_input.T.rename(columns={0: "Value"}))

# === Tab 2: Revenue Predictor ===
with tab2:
    st.title("💵 OEP Revenue Predictor")

    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox("📆 Month", range(1, 13), format_func=lambda m: datetime(2024, m, 1).strftime("%B"), key="rev_month")
        day_of_week = st.selectbox("🗓️ Day of Week", list(range(7)), format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d], key="rev_day")

    with col2:
        precipitation = st.slider("🌧️ Precipitation (inches)", 0.0, 3.0, 0.0, 0.1, key="rev_precip")
        school_break = st.radio("🏫 Is School Out?", [0, 1], format_func=lambda x: "Yes" if x else "No", key="rev_school")

    is_weekend = 1 if day_of_week in [5, 6] else 0

    revenue_input = pd.DataFrame([{
        "Month": month,
        "Precipitation": precipitation,
        "SchoolBreak": school_break,
        "DayOfWeek": day_of_week,
        "IsWeekend": is_weekend
    }])

    # Predict visitors
    predicted_visitors = int(visitor_model.predict(revenue_input)[0])

    # Predict revenue per visitor
    rppv_input = revenue_input[revenue_features]
    predicted_rppv = float(revenue_model.predict(rppv_input)[0])

    # Total revenue = visitors × RPPV
    predicted_total_revenue = predicted_visitors * predicted_rppv

    st.metric("👥 Predicted Visitors", f"{predicted_visitors:,}")
    st.metric("💵 OEP Revenue per Visitor", f"${predicted_rppv:.2f}")
    st.metric("🎯 Total Predicted Revenue", f"${predicted_total_revenue:,.2f}")

    with st.expander("🔍 View Input Data"):
        st.dataframe(revenue_input.T.rename(columns={0: "Value"}))
