import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# Load enhanced dataset with rainfall and school break info
df = pd.read_csv("Dezerland_Visitors_Weather_School.csv")
df["Date"] = pd.to_datetime(df["Date"])

# --- Feature Engineering ---
df["Month"] = df["Date"].dt.month
df["Day"] = pd.Categorical(df["Date"].dt.day_name(),
                           categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

# One-hot encode
df = pd.get_dummies(df, columns=["Day", "Month"], drop_first=False)

# Define feature set used in model
feature_cols = ["Monthly_Rainfall_cm", "Is_School_Break"] + \
               [col for col in df.columns if col.startswith("Day_")] + \
               [col for col in df.columns if col.startswith("Month_")]

X = df[feature_cols]
y = df["Visitors"]

# Train Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, random_state=42)
model.fit(X, y)

# --- Streamlit UI ---
st.set_page_config(page_title="Dezerland Visitor Predictor", layout="centered")
st.title("🎢 Dezerland Visitor Predictor")
st.markdown("Predict how many guests will visit Dezerland Park based on the day, month, and weather conditions.")

# --- User Inputs ---
st.subheader("📅 Choose Conditions")
all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
all_months = list(range(1, 13))

day = st.selectbox("Day of the Week", all_days)
month = st.selectbox("Month", all_months, format_func=lambda m: pd.to_datetime(f"2024-{m}-01").strftime('%B'))
rainfall = st.slider("Monthly Rainfall (cm)", min_value=0.0, max_value=25.0, step=0.1)
is_break = st.checkbox("Is School on Break?", value=False)

# --- Build input vector ---
day_columns = [f"Day_{d}" for d in all_days]
month_columns = [f"Month_{m}" for m in all_months]
input_data = {col: 0 for col in feature_cols}
input_data["Monthly_Rainfall_cm"] = rainfall
input_data["Is_School_Break"] = is_break
input_data[f"Day_{day}"] = 1
input_data[f"Month_{month}"] = 1
input_df = pd.DataFrame([input_data])[feature_cols]

# --- Prediction ---
prediction = model.predict(input_df)[0]
st.metric(label="🎯 Predicted Visitors", value=f"{int(prediction):,}")

# --- Feature Importance ---
st.divider()
st.subheader("📊 What Most Affects Visitor Counts?")
st.caption("Larger values (in either direction) have more predictive power.")

importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.index = importances.index.str.replace("Day_", "Day: ").str.replace("Month_", "Month: ")
st.bar_chart(importances.sort_values())

# --- Historical Trend ---
st.divider()
st.subheader("📈 Historical Visitor Trends")
st.caption("7-day rolling average of past attendance")

df["7-Day Avg"] = df["Visitors"].rolling(window=7).mean()
st.line_chart(df.set_index("Date")[["7-Day Avg"]])
