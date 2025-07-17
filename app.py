import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- SIDEBAR ---
st.sidebar.title("💧 Water Potability Checker")
st.sidebar.write("Enter values below to test water quality")

ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness", 50, 300, 150)
solids = st.sidebar.slider("Solids", 1000, 50000, 10000)
chloramines = st.sidebar.slider("Chloramines", 0.0, 15.0, 7.0)
sulfate = st.sidebar.slider("Sulfate", 100.0, 500.0, 250.0)
conductivity = st.sidebar.slider("Conductivity", 100.0, 1000.0, 500.0)
organic_carbon = st.sidebar.slider("Organic Carbon", 2.0, 30.0, 15.0)
trihalo = st.sidebar.slider("Trihalomethanes", 0.0, 120.0, 60.0)
turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 4.0)

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

# --- MAIN LAYOUT ---
st.title("🚰 Clean Water Prediction &  Chatbot")

# Prediction Section
st.subheader("🧪 Water Potability Prediction")

if st.button("Check Water Quality"):
    prediction = model.predict(input_data)[0]
    result = "✅ Safe to Drink" if prediction == 1 else "⚠️ Not Safe to Drink"
    st.success(result)

# Confusion Matrix
st.subheader("📊 Model Confusion Matrix")
cm_image = "confusion_matrix.png"
st.image(cm_image, caption="Confusion Matrix (Random Forest)")

# Chatbot Section
st.subheader("🤖 Ask about Clean Water & Sanitation (SDG 6)")
query = st.text_input("Ask your question about water quality or SDG 6")

if query:
    with st.spinner("Thinking..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in SDG 6 (Clean Water & Sanitation) and water quality."},
                {"role": "user", "content": query}
            ]
        )
        reply = response.choices[0].message.content
        st.info(reply)

# Visualization Section
st.subheader("📈 Dataset Preview & Visualization")
st.write(df.head())

# Feature Distribution Plot
st.markdown("### Distribution of Safe vs Unsafe Water")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Potability", palette="Set2")
ax.set_xticklabels(["Unsafe", "Safe"])
st.pyplot(fig)
