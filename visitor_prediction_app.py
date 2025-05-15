import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load cleaned data
df = pd.read_csv("Dezerland_Daily_Visitors_Cleaned.csv")

# --- Ensure all days are represented consistently ---
all_days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
df['Day'] = pd.Categorical(df['Day'], categories=all_days, ordered=False)
df = pd.get_dummies(df, columns=['Day'], drop_first=False)

# Feature columns
feature_cols = ['Rainfall', 'Is_Event_Day'] + [col for col in df.columns if col.startswith('Day_')]
X = df[feature_cols]
y = df['Visitors']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit app
st.set_page_config(page_title="Dezerland Visitor Predictor", layout="centered")
st.title("🎢 Dezerland Daily Visitor Predictor")

st.markdown("Predict how many guests will visit based on day, rain, and events.")

# --- User Inputs ---
day = st.selectbox("Day of Week", all_days)
rainfall = st.slider("Rainfall (inches)", min_value=0.0, max_value=2.0, step=0.1)
event = st.checkbox("Special Event Day")

# --- Build consistent input feature set ---
day_columns = [f"Day_{d}" for d in all_days]
input_data = {col: 0 for col in day_columns}
input_data['Rainfall'] = rainfall
input_data['Is_Event_Day'] = 1 if event else 0

# Set appropriate one-hot encoded day
input_data[f"Day_{day}"] = 1

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]  # 🔑 Ensure same column order

# Predict
prediction = model.predict(input_df)[0]
st.metric(label="🎯 Predicted Visitors", value=f"{int(prediction):,}")

# Optional: Show feature importances
if st.checkbox("Show Feature Importances"):
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    st.bar_chart(importances)

# Optional: Show historical data
if st.checkbox("Show Historical Visitor Data"):
    st.line_chart(df[['Visitors']])
