import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load cleaned data
df = pd.read_csv("Dezerland_Daily_Visitors_Cleaned.csv")

# --- Ensure all months and days are represented consistently ---
df['Month'] = pd.to_datetime(df['Date']).dt.month
all_months = list(range(1, 13))
df['Month'] = pd.Categorical(df['Month'], categories=all_months, ordered=False)
df = pd.get_dummies(df, columns=['Month'], drop_first=False)

all_days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
df['Day'] = pd.Categorical(df['Day'], categories=all_days, ordered=False)
df = pd.get_dummies(df, columns=['Day'], drop_first=False)

# Feature columns
feature_cols = ['Rainfall', 'Is_Event_Day'] + \
               [col for col in df.columns if col.startswith('Day_')] + \
               [col for col in df.columns if col.startswith('Month_')]
X = df[feature_cols]
y = df['Visitors']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.set_page_config(page_title="Dezerland Visitor Predictor", layout="centered")
st.title("🎢 Dezerland Daily Visitor Predictor")

st.markdown("Predict how many guests will visit based on day, month, rain, and events.")

# --- User Inputs ---
day = st.selectbox("Day of Week", all_days)
month = st.selectbox("Month", all_months)
rainfall = st.slider("Rainfall (inches)", min_value=0.0, max_value=2.0, step=0.1)
event = st.checkbox("Special Event Day")

# --- Build consistent input feature set ---
day_columns = [f"Day_{d}" for d in all_days]
month_columns = [f"Month_{m}" for m in all_months]
input_data = {col: 0 for col in day_columns + month_columns}
input_data['Rainfall'] = rainfall
input_data['Is_Event_Day'] = 1 if event else 0
input_data[f"Day_{day}"] = 1
input_data[f"Month_{month}"] = 1

input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]  # Ensure correct column order

# Predict
prediction = model.predict(input_df)[0]
st.metric(label="🎯 Predicted Visitors", value=f"{int(prediction):,}")

# Optional: Show feature importances (coefficients)
if st.checkbox("Show Feature Coefficients"):
    importances = pd.Series(model.coef_, index=X.columns).sort_values()
    st.bar_chart(importances)

# Optional: Show historical data
if st.checkbox("Show Historical Visitor Data"):
    st.line_chart(df[['Visitors']])