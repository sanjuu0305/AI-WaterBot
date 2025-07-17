# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openai import OpenAI

# Load model
model = joblib.load("water_model.pkl")

# Streamlit settings
st.set_page_config(page_title="💧 Water Potability Predictor", layout="centered")
st.title("💧  Clean Water & Sanitation")
st.subheader("Check if your water is safe to drink")

st.markdown("Enter water quality values below:")

# Input sliders
ph = st.slider("pH Level", 0.0, 14.0, 7.0)
hardness = st.number_input("Hardness (mg/L)", min_value=50.0, max_value=500.0, value=150.0)
solids = st.number_input("Solids (ppm)", min_value=100.0, max_value=50000.0, value=10000.0)
chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.0)
sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 300.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=100.0, max_value=1000.0, value=450.0)
organic_carbon = st.slider("Organic Carbon (ppm)", 0.0, 30.0, 10.0)
trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 120.0, 60.0)
turbidity = st.slider("Turbidity (NTU)", 0.0, 10.0, 3.0)

input_data = pd.DataFrame([{
    "ph": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes,
    "Turbidity": turbidity
}])

if st.button("🔍 Predict Water Potability"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]

    if prediction == 1:
        st.success(f"✅ Safe to Drink (Confidence: {proba:.2%})")
    else:
        st.error(f"⚠️ Not Safe to Drink (Confidence: {proba:.2%})")

# CSV Upload
st.markdown("---")
st.write("📄 Upload a CSV to predict multiple samples:")
csv_file = st.file_uploader("Upload CSV", type=["csv"])

if csv_file:
    uploaded_data = pd.read_csv(csv_file)
    uploaded_data.fillna(uploaded_data.mean(), inplace=True)
    predictions = model.predict(uploaded_data)
    uploaded_data["Prediction"] = predictions
    uploaded_data["Result"] = uploaded_data["Prediction"].map({0: "Unsafe", 1: "Safe"})
    st.write("Prediction Results:")
    st.dataframe(uploaded_data)
    csv = uploaded_data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv, "water_potability (1).csv", "text/csv")

# ------------------------------------------------------------------
# 🧠 ChatGPT-Style Assistant for SDG 6
# ------------------------------------------------------------------

st.markdown("---")
st.subheader("💬 Ask Our Water & Sanitation Assistant")

# API key (from Streamlit Secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Setup conversation memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant for Clean Water and Sanitation. Answer questions about water quality, sanitation, sustainability, and related innovations."}
    ]

# User input
user_query = st.chat_input("Ask about water, sanitation, potability...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state["messages"]
    )

    bot_reply = response.choices[0].message.content
    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

# Chat UI
for msg in st.session_state["messages"][1:]:  # Skip system
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
