# app.py
import streamlit as st
import pandas as pd
import requests
import time
import os
import plotly.express as px
import urllib.parse
from typing import Dict, Any

st.set_page_config(page_title="CleanWater Predictor", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers for Lottie animations (no extra pip packages required) -----------------
def lottie_html(lottie_url: str, height: int = 250):
    """
    Returns HTML string embedding the lottie-player for a given public Lottie JSON URL.
    Uses CDNs, no additional Streamlit packages required.
    """
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{lottie_url}"  background="transparent"  speed="1"  style="width:100%; height:{height}px;"  loop  autoplay></lottie-player>
    """
    return html

# Small styling
st.markdown(
    """
    <style>
    .stApp { background-color: #f8fbff; }
    .header-emoji { font-size: 1.3rem; }
    .small-note { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- HEADER -------------------
col1, col2 = st.columns([2, 1])
with col1:
    st.title("💧 AI-based Clean Water Predictor")
    st.markdown("Analyze your water quality using sensor data and an explainable AI heuristic. Predict safety, visualize results, chat with WaterBot, and share findings.")
with col2:
    # Lottie animation (public lottie json link). You can replace the URL with any Lottie JSON link.
    lottie_url_header = "https://assets6.lottiefiles.com/packages/lf20_j1adxtyb.json"
    st.components.v1.html(lottie_html(lottie_url_header, height=160), height=170)

st.markdown("---")

# ------------------- SIDEBAR NAV -------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Predictor", "Visualize", "Chatbot", "About"])

st.sidebar.markdown("---")
use_remote = st.sidebar.checkbox("Use remote API", value=False)
remote_url = st.sidebar.text_input("Remote API URL", value="") if use_remote else ""

# Updated labels to reflect Google Gemini usage
use_llm = st.sidebar.checkbox("Enable Gemini (Google AI)", value=False)
api_key = st.sidebar.text_input("Google AI Studio API Key", type="password") if use_llm else None

# ------------------- PREDICTOR LOGIC -------------------
def local_predict_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple explainable rule-based risk estimator. Accepts dict-like row with keys:
    pH / ph, tds / TDS, turbidity / turb, temp / temperature
    """
    try:
        ph = float(row.get("pH", row.get("ph", 7.0)))
    except Exception:
        ph = 7.0
    try:
        tds = float(row.get("tds", row.get("TDS", 0.0)))
    except Exception:
        tds = 0.0
    try:
        turb = float(row.get("turbidity", row.get("turb", 0.0)))
    except Exception:
        turb = 0.0
    try:
        temp = float(row.get("temp", row.get("temperature", 25.0)))
    except Exception:
        temp = 25.0

    score = 0.0
    reasons = []

    # pH rules
    if ph < 6.5:
        score += 0.2
        reasons.append(f"Low pH ({ph})")
    elif ph > 8.5:
        score += 0.15
        reasons.append(f"High pH ({ph})")

    # TDS rules
    if tds > 1000:
        score += 0.4
        reasons.append(f"Very high TDS ({tds} mg/L)")
    elif tds > 500:
        score += 0.25
        reasons.append(f"High TDS ({tds} mg/L)")
    elif tds > 300:
        score += 0.1
        reasons.append(f"Moderate TDS ({tds} mg/L)")

    # Turbidity & temp
    if turb > 5:
        score += 0.2
        reasons.append(f"High turbidity ({turb} NTU)")
    if temp > 35:
        score += 0.05
        reasons.append(f"High temperature ({temp}°C)")

    risk_score = min(1.0, score)
    prediction = "Unsafe 🚫" if risk_score >= 0.5 else "Safe ✅"
    action = "Avoid drinking; boil/filter & retest. Seek lab analysis for confirmation." if prediction.startswith("Unsafe") else "Water appears likely safe; use normal precautions (boil if uncertain)."

    return {
        "prediction": prediction,
        "risk_score": round(risk_score, 3),
        "reasons": reasons or ["No major issues detected"],
        "action": action,
        "pH": ph,
        "tds": tds,
        "turbidity": turb,
        "temp": temp,
    }

def call_remote_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not remote_url:
        return {"error": "Remote URL not provided."}
    try:
        r = requests.post(remote_url, json=payload, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def show_result(res: Dict[str, Any]):
    if not isinstance(res, dict):
        st.error("Invalid response from predictor.")
        return
    if "error" in res:
        st.error("Error: " + str(res["error"]))
        return
    st.success(f"Prediction: {res.get('prediction', 'N/A')}  |  Risk Score: {res.get('risk_score', 'N/A')}")
    st.markdown("**Recommended Action:** " + str(res.get("action", "")))
    st.markdown("**Reasons:**")
    for r in res.get("reasons", ["No data"]):
        st.write(f"- {r}")

# Ensure session state keys exist
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

# ------------------- PREDICTOR PAGE -------------------
if page == "Predictor":
    st.subheader("🔬 Single Sample Prediction")
    # Tiny animation
    st.components.v1.html(lottie_html("https://assets6.lottiefiles.com/packages/lf20_er8p8l5o.json", height=120), height=130)

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
        payload = {"pH": ph, "tds": tds, "turbidity": turbidity, "temp": temp, "notes": notes}
        if use_remote and remote_url:
            res = call_remote_api(payload)
        else:
            res = local_predict_row(payload)
        show_result(res)

        # Tweet link (properly encoded)
        tweet_text = f"AI CleanWater Predictor result: {res.get('prediction','N/A')} (Risk {res.get('risk_score','N/A')}). #AI #WaterQuality"
        tweet_url = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(tweet_text)
        st.markdown(f"[🐦 Share on Twitter]({tweet_url})", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📂 Batch Upload (CSV)")
    uploaded = st.file_uploader("Upload CSV with columns: pH, tds, turbidity, temp (headers not case-sensitive)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None

        if df is not None:
            st.dataframe(df.head())
            if st.button("Predict Batch"):
                results = []
                for _, row in df.iterrows():
                    # convert row to dict and call local_predict_row
                    res = local_predict_row(row.to_dict())
                    # merge the original row (to preserve other columns) with prediction fields
                    merged = {**row.to_dict(), **res}
                    results.append(merged)
                out = pd.DataFrame(results)
                st.session_state["batch_results"] = out
                st.success("Batch prediction completed!")
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download Results CSV", csv, "batch_predictions.csv", "text/csv")

# ------------------- VISUALIZATION PAGE -------------------
elif page == "Visualize":
    st.subheader("📊 Data Visualization")
    st.components.v1.html(lottie_html("https://assets2.lottiefiles.com/packages/lf20_kq5r7x7q.json", height=120), height=130)

    if st.session_state["batch_results"] is None:
        st.info("Upload and predict batch data on the Predictor page first.")
    else:
        df = st.session_state["batch_results"]
        # Ensure numeric columns exist
        for col in ["risk_score", "pH", "tds", "turbidity", "temp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.write("### Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", nbins=12, title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

        if "pH" in df.columns and "tds" in df.columns:
            st.write("### pH vs TDS (Colored by Prediction)")
            fig2 = px.scatter(df, x="pH", y="tds", color="prediction", size="risk_score",
                              hover_data=["turbidity", "temp"], title="pH vs TDS")
            st.plotly_chart(fig2, use_container_width=True)

        st.write("### Summary Statistics")
        st.dataframe(df.describe(include="all").T)

# ------------------- CHATBOT PAGE -------------------
elif page == "Chatbot":
    st.subheader("💬 WaterBot Assistant")
    st.markdown("Ask questions about water safety, purification, or your readings.")
    st.components.v1.html(lottie_html("https://assets3.lottiefiles.com/packages/lf20_kdx6cani.json", height=120), height=130)

    if use_llm and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            user_input = st.text_input("You:", placeholder="e.g., What does high TDS mean?")
            if st.button("Ask WaterBot") and user_input:
                with st.spinner("WaterBot thinking..."):
                    try:
                        model = genai.GenerativeModel("gemini-1.5-flash-latest")
                        response = model.generate_content(user_input)
                        st.markdown(f"**WaterBot:** {response.text}")
                    except Exception as e:
                        st.error(f"⚠️ Failed to generate response: {e}")
        except Exception as e:
            st.error(f"⚠️ Gemini setup error: {e}")
            st.info("Make sure you've installed `google-generativeai` using `pip install -U google-generativeai`.")
    else:
        st.warning("Enable Gemini (Google AI) and paste your Google AI Studio API key in the sidebar to chat with WaterBot.")
# ------------------- ABOUT PAGE -------------------
elif page == "About":
    st.subheader("🌍 About CleanWater Predictor")
    st.components.v1.html(lottie_html("https://assets4.lottiefiles.com/packages/lf20_uroy2j2l.json", height=140), height=150)
    st.markdown("""
    **CleanWater Predictor** is a student AI project developed during the GTU AI Summer Internship 2025
    in collaboration with IBM SkillsBuild and CSRBOX.

    **Features**
    - Local & API-based predictions
    - Batch CSV analysis and downloadable results
    - Visual dashboards
    - AI-powered chatbot (WaterBot)
    - Twitter sharing integration

    **Team:** WaterBot Project | GTU 2025
    """)
    st.markdown("**Notes & disclaimers:** This app uses a simple rule-based estimator for demonstration and educational purposes. Always confirm with certified lab testing before making health-critical decisions.")