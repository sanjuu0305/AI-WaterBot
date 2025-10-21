# app.py (Enhanced Version)
import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.express as px

st.set_page_config(page_title="CleanWater Predictor", layout="wide")

# ------------------- HEADER -------------------
st.title("💧 AI-based Clean Water Predictor")
st.markdown("""
Analyze your water quality using sensor data and AI-based prediction.  
Predict safety, visualize results, chat with WaterBot, and share your findings.
""")

# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "Visualize", "Chatbot", "About"])

st.sidebar.markdown("---")
use_remote = st.sidebar.checkbox("Use remote API", value=False)
remote_url = st.sidebar.text_input("Remote API URL", value="")
use_llm = st.sidebar.checkbox("Enable LLM (OpenAI)", value=False)
openai_key = st.sidebar.text_input("OPENAI API KEY", type="password") if use_llm else None

# ------------------- PREDICTOR PAGE -------------------
def local_predict_row(row):
    ph = float(row.get("pH", row.get("ph", 7.0)))
    tds = float(row.get("tds", row.get("TDS", 0.0)))
    turb = float(row.get("turbidity", row.get("turbidity", 0.0)))
    temp = float(row.get("temp", row.get("temperature", 25.0)))

    score = 0.0
    reasons = []
    if ph < 6.5:
        score += 0.2; reasons.append(f"Low pH ({ph})")
    elif ph > 8.5:
        score += 0.15; reasons.append(f"High pH ({ph})")

    if tds > 1000:
        score += 0.4; reasons.append(f"Very high TDS ({tds} mg/L)")
    elif tds > 500:
        score += 0.25; reasons.append(f"High TDS ({tds} mg/L)")
    elif tds > 300:
        score += 0.1; reasons.append(f"Moderate TDS ({tds} mg/L)")

    if turb > 5:
        score += 0.2; reasons.append(f"High turbidity ({turb} NTU)")
    if temp > 35:
        score += 0.05; reasons.append(f"High temperature ({temp}°C)")

    risk_score = min(1.0, score)
    prediction = "Unsafe 🚫" if risk_score >= 0.5 else "Safe ✅"
    action = "Avoid drinking, boil/filter & retest." if prediction == "Unsafe 🚫" else "Water appears safe; normal precautions advised."

    return {
        "prediction": prediction,
        "risk_score": round(risk_score, 3),
        "reasons": reasons or ["No major issues detected"],
        "action": action
    }

def call_remote_api(payload):
    try:
        r = requests.post(remote_url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def show_result(res):
    if "error" in res:
        st.error("Error: " + res["error"])
        return
    st.success(f"**Prediction:** {res['prediction']}  |  **Risk Score:** {res['risk_score']}")
    st.markdown("**Recommended Action:** " + res["action"])
    st.markdown("**Reasons:**")
    for r in res["reasons"]:
        st.write(f"- {r}")

if page == "Predictor":
    st.subheader("🔬 Single Sample Prediction")
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            tds = st.number_input("TDS (mg/L)", 0.0, 2000.0, 100.0, 1.0)
        with col2:
            turbidity = st.number_input("Turbidity (NTU)", 0.0, 100.0, 1.0, 0.1)
            temp = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, 0.1)
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {"pH": ph, "tds": tds, "turbidity": turbidity, "temp": temp}
        if use_remote and remote_url:
            res = call_remote_api(payload)
        else:
            res = local_predict_row(payload)
        show_result(res)

        # Tweet link
        tweet_text = f"AI CleanWater Predictor result: {res['prediction']} (Risk {res['risk_score']}). #AI #WaterQuality"
        tweet_url = f"https://twitter.com/intent/tweet?text={tweet_text}"
        st.markdown(f"[🐦 Share on Twitter]({tweet_url})")

    st.markdown("---")
    st.subheader("📂 Batch Upload (CSV)")
    uploaded = st.file_uploader("Upload CSV with columns: pH, tds, turbidity, temp", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict Batch"):
            results = []
            for _, row in df.iterrows():
                results.append({**row.to_dict(), **local_predict_row(row)})
            out = pd.DataFrame(results)
            st.session_state["batch_results"] = out
            st.success("Batch prediction completed!")
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download Results CSV", csv, "batch_predictions.csv", "text/csv")

# ------------------- VISUALIZATION PAGE -------------------
elif page == "Visualize":
    st.subheader("📊 Data Visualization")
    if "batch_results" not in st.session_state:
        st.info("Upload and predict batch data first.")
    else:
        df = st.session_state["batch_results"]
        st.write("### Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", nbins=10, title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

        ### FIX START
        import numpy as np
        df = df.rename(columns={"ph": "pH", "TDS": "tds"})
        for col in ["pH", "tds", "risk_score", "turbidity", "temp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["pH", "tds", "risk_score"])
        df["risk_score"] = np.clip(df["risk_score"], 0.01, 1.0)
        ### FIX END

        st.write("### pH vs TDS (Colored by Prediction)")
        fig2 = px.scatter(df, x="pH", y="tds", color="prediction",
                          size="risk_score", hover_data=["turbidity", "temp"])
        st.plotly_chart(fig2, use_container_width=True)

        st.write("### Summary Statistics")
        st.dataframe(df.describe())

# ------------------- CHATBOT PAGE -------------------
elif page == "Chatbot":
    st.subheader("💬 WaterBot Assistant")
    st.markdown("Ask questions about water safety, purification, or your readings.")

    if use_llm and openai_key:
       ### FIX START — Update to new OpenAI client (for openai>=1.0.0)
import OpenAI

client = OpenAI(api_key=openai_key)
user_input = st.text_input("You:", placeholder="e.g., What does high TDS mean?")

if user_input:
    with st.spinner("Thinking..."):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are WaterBot, an expert in water quality."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=300,
            )
            reply = completion.choices[0].message.content
            st.markdown(f"**WaterBot:** {reply}")
        except Exception as e:
            st.error(f"LLM error: {e}")
### FIX END
        st.warning("Enable LLM and provide API key in sidebar to chat with WaterBot.")

# ------------------- ABOUT PAGE -------------------
elif page == "About":
    st.subheader("🌍 About CleanWater Predictor")
    st.markdown("""
**CleanWater Predictor** is a student AI project developed during the GTU AI Summer Internship 2025  
in collaboration with IBM SkillsBuild and CSRBOX.

**Features:**
- Local & API-based predictions  
- Batch CSV analysis  
- Visual dashboards  
- AI-powered chatbot (WaterBot)  
- Twitter sharing integration  

**Team:** WaterBot Project | GTU 2025  
""")