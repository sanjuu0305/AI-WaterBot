import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# Streamlit page config
st.set_page_config(page_title="AI Chatbot & Water Potability", layout="wide")

# Load trained model
model = joblib.load("water_model.pkl")

# Load dataset safely
df = pd.read_csv("water_potability (1).csv")
df.fillna(df.mean(numeric_only=True), inplace=True)

# OpenAI client setup (from secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- SIDEBAR: User Input ---
st.sidebar.title("💧 Potability Checker")
st.sidebar.write("Adjust the sliders below to test your water sample:")

ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness (mg/L)", 50.0, 500.0, 150.0)
solids = st.sidebar.slider("Solids (ppm)", 1000.0, 50000.0, 10000.0)
chloramines = st.sidebar.slider("Chloramines (ppm)", 0.0, 15.0, 7.0)
sulfate = st.sidebar.slider("Sulfate (mg/L)", 100.0, 500.0, 250.0)
conductivity = st.sidebar.slider("Conductivity (μS/cm)", 100.0, 1000.0, 500.0)
organic_carbon = st.sidebar.slider("Organic Carbon (ppm)", 2.0, 30.0, 15.0)
trihalo = st.sidebar.slider("Trihalomethanes (μg/L)", 0.0, 120.0, 60.0)
turbidity = st.sidebar.slider("Turbidity (NTU)", 0.0, 10.0, 4.0)

input_data = pd.DataFrame([{
    "ph": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalo,
    "Turbidity": turbidity
}])

# --- MAIN UI ---
st.title("🚰 Clean Water AI Model + Chatbot Assistant")

# Prediction Output
st.subheader("🧪 Water Potability Prediction")

if st.button("🔍 Check Water Quality"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][prediction]
    if prediction == 1:
        st.success(f"✅ Safe to Drink (Confidence: {prob:.2%})")
    else:
        st.error(f"⚠️ Not Safe to Drink (Confidence: {prob:.2%})")

# Confusion Matrix Image
st.subheader("📊 Confusion Matrix")
st.image("confusion_matrix.png", caption="Random Forest Classifier Results", use_container_width=True)

# --- AI CHATBOT ---
st.subheader("🤖 Ask AI about Water Quality, and Sanitation")

if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "system", "content": "You are an AI assistant specializing in Clean Water and Sanitation. Provide concise, informative answers on water quality, pollution, sanitation practices, and global sustainability."},
        {"role": "user", "content": query}
    ]

user_input = st.text_input("💬 Ask your question:")
if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = openai.ChatCompletions.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.chat
        )
        reply = response.choices[0].message.content
        st.session_state.chat.append({"role": "assistant", "content": reply})
        st.success(reply)

# --- VISUALIZATION ---
st.subheader("📈 Dataset Overview & Visualization")

# Preview data
st.markdown("### Sample of Water Dataset")
st.dataframe(df.head())

# Distribution plot
st.markdown("### Potability Distribution")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Potability", palette="Set2")
ax.set_xticklabels(["Not Potable", "Potable"])
plt.title("Safe vs Unsafe Water Samples")
st.pyplot(fig)
