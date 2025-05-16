import streamlit as st
import pandas as pd

# Load and preprocess raw data
df = pd.read_csv("Dezerland_Visitors_Weather_School.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["DayName"] = df["Date"].dt.day_name()

# Compute average visitors per (Month, DayName)
avg_lookup = df.groupby(["Month", "DayName"])["Visitors"].mean().reset_index()

# --- Streamlit UI ---
st.set_page_config(page_title="Dezerland Visitor Predictor (Historical Average)", layout="centered")
st.title("🎢 Dezerland Visitor Predictor (Historical Average)")
st.markdown("Predict expected visitor count based on actual historical averages.")

# --- User Inputs ---
st.subheader("📅 Choose Day")
all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
all_months = list(range(1, 13))

month = st.selectbox("Month", all_months, format_func=lambda m: pd.to_datetime(f"2024-{m}-01").strftime('%B'))
day = st.selectbox("Day of the Week", all_days)

# --- Predict from Lookup Table ---
match = avg_lookup[(avg_lookup["Month"] == month) & (avg_lookup["DayName"] == day)]

if not match.empty:
    prediction = int(match["Visitors"].values[0])
    st.metric(label="🎯 Predicted Visitors (Historical Average)", value=f"{prediction:,}")
else:
    st.warning("No historical data found for that combination.")

# --- Historical Chart ---
st.divider()
st.subheader("📈 Monthly Visitor Totals")
df["MonthPeriod"] = df["Date"].dt.to_period("M")
monthly_summary = df.groupby("MonthPeriod")["Visitors"].sum()
monthly_summary.index = monthly_summary.index.astype(str)
st.bar_chart(monthly_summary)

# --- Optional: Show full lookup table ---
st.divider()
with st.expander("📊 Show Full Monthly-Day Visitor Averages"):
    st.dataframe(avg_lookup.sort_values(["Month", "DayName"]))
