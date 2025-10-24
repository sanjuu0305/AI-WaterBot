import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import urllib.parse
from typing import Dict, Any

# Page setup
st.set_page_config(page_title="CleanWater Predictor", layout="wide", initial_sidebar_state="expanded")

# ----------------- Helpers for Lottie animations -----------------
def lottie_html(lottie_url: str, height: int = 250):
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{lottie_url}" background="transparent" speed="1"
    style="width:100%; height:{height}px;" loop autoplay></lottie-player>
    """
    return html


# ----------------- Custom Styling -----------------
st.markdown("""
<style>  
.stApp {  
    background: linear-gradient(180deg, #e0f7fa, #f8fbff);  
    background-attachment: fixed;  
    color: #1e293b;  
}  

@keyframes fadeIn {  
    from {opacity: 0; transform: translateY(15px);}  
    to {opacity: 1; transform: translateY(0);}  
}  
.fade-section { animation: fadeIn 1.2s ease-in-out; }  

html { scroll-behavior: smooth; }  

.sidebar-title {  
    font-size: 1.2rem;  
    font-weight: 600;  
    color: #0077b6;  
    margin-bottom: 8px;  
}  

div[role='radiogroup'] label {  
    font-size: 1.1rem !important;  
    padding: 8px 12px;  
    transition: all 0.3s ease;  
}  
div[role='radiogroup'] label:hover {  
    background-color: #c7f9cc;  
    border-radius: 10px;  
}  

.stButton>button {  
    background: linear-gradient(90deg, #0077b6, #48cae4);  
    color: white;  
    border-radius: 10px;  
    border: none;  
    padding: 0.6rem 1rem;  
    transition: all 0.3s ease-in-out;  
}  
.stButton>button:hover {  
    transform: scale(1.05);  
    background: linear-gradient(90deg, #48cae4, #0077b6);  
}  

h1, h2, h3 { text-shadow: 1px 1px 2px #a0e7e5; }  
</style>  
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown("<div class='fade-section'>", unsafe_allow_html=True)
col1, col2 = st.columns([2, 1])
with col1:
    st.title("💧 AI-based Clean Water Predictor")
    st.markdown("Analyze water quality using AI — predict safety, visualize results, chat with WaterBot, and share findings.")
with col2:
    st.components.v1.html(lottie_html("https://assets4.lottiefiles.com/packages/lf20_uroy2j2l.json", 160), height=170)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("---")


# ---------------- SIDEBAR NAV ----------------
st.sidebar.markdown("<div class='sidebar-title'>✨ Navigation</div>", unsafe_allow_html=True)
page = st.sidebar.radio("Select Page", ["🔬 Predictor", "📊 Visualize", "🤖 Chatbot", "ℹ️ About"], key="nav")

st.sidebar.markdown("---")
use_remote = st.sidebar.checkbox("Use remote API", value=False)
remote_url = st.sidebar.text_input("🌐 Remote API URL", value="") if use_remote else ""

use_llm = st.sidebar.checkbox("🤖 Enable Gemini (Google AI)", value=False)
api_key = st.sidebar.text_input("🔑 Google AI Studio API Key", type="password") if use_llm else None


# ---------------- PREDICTOR LOGIC ----------------
def local_predict_row(row: Dict[str, Any]) -> Dict[str, Any]:
    try: ph = float(row.get("pH", 7.0))
    except Exception: ph = 7.0
    try: tds = float(row.get("tds", 0.0))
    except Exception: tds = 0.0
    try: turb = float(row.get("turbidity", 0.0))
    except Exception: turb = 0.0
    try: temp = float(row.get("temp", 25.0))
    except Exception: temp = 25.0

    score, reasons = 0.0, []
    if ph < 6.5: score += 0.2; reasons.append(f"Low pH ({ph})")
    elif ph > 8.5: score += 0.15; reasons.append(f"High pH ({ph})")
    if tds > 1000: score += 0.4; reasons.append(f"Very high TDS ({tds} mg/L)")
    elif tds > 500: score += 0.25; reasons.append(f"High TDS ({tds} mg/L)")
    elif tds > 300: score += 0.1; reasons.append(f"Moderate TDS ({tds} mg/L)")
    if turb > 5: score += 0.2; reasons.append(f"High turbidity ({turb} NTU)")
    if temp > 35: score += 0.05; reasons.append(f"High temperature ({temp}°C)")

    risk_score = min(1.0, score)
    prediction = "Unsafe 🚫" if risk_score >= 0.5 else "Safe ✅"
    action = "Avoid drinking; boil/filter & retest." if prediction.startswith("Unsafe") else "Water seems fine; still boil if unsure."

    return {
        "prediction": prediction,
        "risk_score": round(risk_score, 3),
        "reasons": reasons or ["No major issues detected"],
        "action": action,
        "pH": ph, "tds": tds, "turbidity": turb, "temp": temp
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
        st.error("Invalid response.")
        return
    if "error" in res:
        st.error("Error: " + str(res["error"]))
        return
    st.success(f"Prediction: {res.get('prediction')} | Risk Score: {res.get('risk_score')}")
    st.markdown("Recommended Action: " + res.get("action", ""))
    st.markdown("Reasons:")
    for r in res.get("reasons", ["No data"]):
        st.write(f"- {r}")


if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None


# ---------------- PAGE CONTENT ----------------
if "Predictor" in page:
    st.markdown("<div class='fade-section'>", unsafe_allow_html=True)
    st.subheader("🔬 Single Sample Prediction")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
            tds = st.number_input("TDS (mg/L)", 0.0, 2000.0, 100.0, 1.0)
        with col2:
            turbidity = st.number_input("Turbidity (NTU)", 0.0, 100.0, 1.0, 0.1)
            temp = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0, 0.1)
        notes = st.text_area("📝 Notes (optional)")
        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        payload = {"pH": ph, "tds": tds, "turbidity": turbidity, "temp": temp, "notes": notes}
        res = call_remote_api(payload) if use_remote and remote_url else local_predict_row(payload)
        show_result(res)
        tweet_text = f"AI CleanWater Predictor result: {res.get('prediction')} (Risk {res.get('risk_score')}). #AI #WaterQuality"
        tweet_url = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(tweet_text)
        st.markdown(f"[🐦 Share on Twitter]({tweet_url})")

    st.markdown("---")
    st.subheader("📂 Batch Upload (CSV)")
    uploaded = st.file_uploader("Upload CSV with columns: pH, tds, turbidity, temp", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            if st.button("🧮 Predict Batch"):
                results = [{**row.to_dict(), **local_predict_row(row.to_dict())} for _, row in df.iterrows()]
                out = pd.DataFrame(results)
                st.session_state["batch_results"] = out
                st.success("✅ Batch prediction completed!")
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results CSV", csv, "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

elif "Visualize" in page:
    st.markdown("<div class='fade-section'>", unsafe_allow_html=True)
    st.subheader("📊 Data Visualization")

    if st.session_state["batch_results"] is None:
        st.info("Upload and predict batch data first.")
    else:
        df = st.session_state["batch_results"]
        for col in ["risk_score", "pH", "tds", "turbidity", "temp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        st.plotly_chart(px.histogram(df, x="risk_score", nbins=12, title="Risk Score Distribution"), use_container_width=True)
        if "pH" in df.columns and "tds" in df.columns:
            fig2 = px.scatter(df, x="pH", y="tds", color="prediction", size="risk_score",
                              hover_data=["turbidity", "temp"], title="pH vs TDS")
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df.describe(include="all").T)
    st.markdown("</div>", unsafe_allow_html=True)

elif "Chatbot" in page:
    st.markdown("<div class='fade-section'>", unsafe_allow_html=True)
    st.subheader("💬 WaterBot Assistant")

    from streamlit_lottie import st_lottie
    import json

    def load_lottiefile(filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    lottie_chatbot = load_lottiefile("Live chatbot (3).json")
    st_lottie(lottie_chatbot, height=200, key="chatbot_lottie")

    if use_llm and api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            user_input = st.text_input("You:", placeholder="💭 Ask about water safety...")
            if st.button("🤖 Ask WaterBot") and user_input:
                with st.spinner("WaterBot thinking..."):
                    model = genai.GenerativeModel("gemini-1.5-flash-latest")
                    response = model.generate_content(user_input)
                    st.markdown(f"**WaterBot:** {response.text}")
        except Exception as e:
            st.error(f"⚠️ Gemini setup error: {e}")
            st.info("Make sure you've installed `google-generativeai` via `pip install -U google-generativeai`.")
    else:
        st.warning("Enable Gemini and paste your API key to chat with WaterBot.")
    st.markdown("</div>", unsafe_allow_html=True)

elif "About" in page:
    st.markdown("<div class='fade-section'>", unsafe_allow_html=True)
    st.subheader("🌍 About CleanWater Predictor")
    st.markdown("""
    **CleanWater Predictor** — built during the GTU AI Summer Internship 2025  
    in collaboration with IBM SkillsBuild and CSRBOX.  

    **✨ Features**
    - Local & API-based predictions  
    - Batch CSV analysis  
    - Visual dashboards  
    - AI chatbot (WaterBot)  
    - Twitter sharing integration  

    **Team:** WaterBot Project | GTU 2025  

    _Disclaimer: This app uses a rule-based estimator for demo purposes. Always confirm results with lab testing._
    """)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- Scroll-triggered animation for Lotties ----------------
st.markdown("""
<script>
const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = 1;
      entry.target.style.transform = "translateY(0)";
    }
  });
});
document.querySelectorAll("lottie-player").forEach(el => {
  el.style.opacity = 0;
  el.style.transform = "translateY(20px)";
  el.style.transition = "all 1s ease";
  observer.observe(el);
});
</script>
""", unsafe_allow_html=True)