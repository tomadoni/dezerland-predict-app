import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load cleaned data
df = pd.read_csv("Dezerland_Daily_Visitors_Cleaned.csv")

# --- Prepare categorical features ---
df['Month'] = pd.to_datetime(df['Date']).dt.month
all_months = list(range(1, 13))
df['Month'] = pd.Categorical(df['Month'], categories=all_months, ordered=False)
df = pd.get_dummies(df, columns=['Month'], drop_first=False)

all_days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
df['Day'] = pd.Categorical(df['Day'], categories=all_days, ordered=False)
df = pd.get_dummies(df, columns=['Day'], drop_first=False)

# Feature columns (no event day)
feature_cols = ['Rainfall'] + \
               [col for col in df.columns if col.startswith('Day_')] + \
               [col for col in df.columns if col.startswith('Month_')]
X = df[feature_cols]
y = df['Visitors']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI setup
st.set_page_config(page_title="Dezerland Visitor Predictor", layout="centered")
st.title("🎢 Dezerland Visitor Predictor")

st.markdown("Predict how many guests will visit Dezerland Park based on the day, month, and weather conditions.")

# --- User Inputs ---
st.subheader("📅 Choose Conditions")
day = st.selectbox("Day of the Week", all_days)
month = st.selectbox("Month", all_months)
rainfall = st.slider("Rainfall (inches)", min_value=0.0, max_value=2.0, step=0.1)

# --- Build input feature row ---
day_columns = [f"Day_{d}" for d in all_days]
month_columns = [f"Month_{m}" for m in all_months]
input_data = {col: 0 for col in day_columns + month_columns}
input_data['Rainfall'] = rainfall
input_data[f"Day_{day}"] = 1
input_data[f"Month_{month}"] = 1
input_df = pd.DataFrame([input_data])
input_df = input_df[X.columns]  # Ensure order

# --- Prediction ---
prediction = model.predict(input_df)[0]
st.metric(label="🎯 Predicted Visitors", value=f"{int(prediction):,}")

# --- Feature Importance / Coefficients ---
st.divider()
st.subheader("📊 What Most Affects Visitor Counts?")
st.caption("Positive values increase expected visitors, negative values reduce them.")

importances = pd.Series(model.coef_, index=X.columns)
importances.index = importances.index.str.replace("Day_", "Day: ").str.replace("Month_", "Month: ")
st.bar_chart(importances.sort_values())

# --- Historical Trend View ---
st.divider()
st.subheader("📈 Historical Visitor Trends")
st.caption("7-day rolling average of past attendance")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date")
    df['7-Day Avg'] = df['Visitors'].rolling(window=7).mean()
    st.line_chart(df.set_index("Date")[['7-Day Avg']])
else:
    st.write("Date column missing from data.")