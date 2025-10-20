# app.py
import streamlit as st
import pandas as pd
import requests
import time
import os

st.set_page_config(page_title="CleanWater Predictor", layout="centered")

st.title("💧 AI-based Clean Water Predictor (Streamlit)")
st.markdown("Enter sensor readings (pH, TDS, Turbidity, Temp). Use **Predict** to get a risk score and recommended action.")

# --- Sidebar settings ---
st.sidebar.header("Settings")
use_remote = st.sidebar.checkbox("Call remote prediction API instead of local logic", value=False)
remote_url = st.sidebar.text_input("Remote API URL (POST /predict)", value="", help="e.g. https://my-backend.example.com/predict")
use_llm = st.sidebar.checkbox("Generate explanation using LLM (OpenAI)", value=False)
openai_key = st.sidebar.text_input("OPENAI API KEY (for LLM)", type="password") if use_llm else None

st.sidebar.markdown("---")
st.sidebar.markdown("Deploy notes: Put your keys into Streamlit Secrets (recommended).")

# --- Input form ---
with st.form("sensor_form"):
    st.subheader("Single sample")
    c1, c2 = st.columns(2)
    with c1:
        ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        tds = st.number_input("TDS (mg/L)", min_value=0.0, value=100.0, step=1.0)
    with c2:
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=1.0, step=0.1)
        temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.1)

    lat = st.text_input("Latitude (optional)", value="")
    lon = st.text_input("Longitude (optional)", value="")
    notes = st.text_area("Notes (optional)", value="", height=80)

    submitted = st.form_submit_button("Predict")

# --- Batch upload ---
st.markdown("---")
st.subheader("Batch upload (CSV)")
st.markdown("CSV must contain columns: pH, tds, turbidity, temp, (optional) lat, lon, notes")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Predict batch"):
            st.session_state["batch_df"] = df
    except Exception as e:
        st.error("Can't read CSV: " + str(e))

# --- Helper functions ---
def local_predict_row(row):
    """
    Simple threshold-based predictor. Replace with model call / ONNX inference or remote API.
    Returns dict with prediction, risk_score (0-1), reasons, action.
    """
    ph = float(row.get("pH", row.get("ph", 7.0)))
    tds = float(row.get("tds", row.get("TDS", 0.0)))
    turb = float(row.get("turbidity", row.get("turbidity", 0.0)))
    temp = float(row.get("temp", row.get("temperature", 25.0)))

    score = 0.0
    reasons = []

    # pH: safe range 6.5 - 8.5 (example)
    if ph < 6.5:
        score += 0.2
        reasons.append(f"Low pH ({ph})")
    elif ph > 8.5:
        score += 0.15
        reasons.append(f"High pH ({ph})")

    # TDS thresholds (example)
    if tds > 1000:
        score += 0.4
        reasons.append(f"Very high TDS ({tds} mg/L)")
    elif tds > 500:
        score += 0.25
        reasons.append(f"High TDS ({tds} mg/L)")
    elif tds > 300:
        score += 0.1
        reasons.append(f"Moderate TDS ({tds} mg/L)")

    # Turbidity
    if turb > 5:
        score += 0.2
        reasons.append(f"High turbidity ({turb} NTU)")

    # Temperature (affects microbial growth) — small effect
    if temp > 35:
        score += 0.05
        reasons.append(f"High temperature ({temp}°C)")

    # clamp score
    risk_score = min(1.0, score)
    prediction = "unsafe" if risk_score >= 0.5 else "safe"
    action = "If unsafe: avoid drinking, boil or use certified filter, and perform lab test." if prediction == "unsafe" else "Water appears safe; follow regular precautions."

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

# --- Run prediction ---
def show_result(res, debug_payload=None):
    if "error" in res:
        st.error("Prediction API error: " + res["error"])
        return

    st.markdown("### Result")
    st.write("**Prediction:**", res.get("prediction", "unknown"))
    st.write("**Risk score:**", res.get("risk_score", "n/a"))
    st.write("**Recommended action:**", res.get("recommended_action_short", res.get("action", "")))
    st.write("**Reasons:**")
    for r in res.get("reasons", []):
        if isinstance(r, dict):
            st.write(f"- {r.get('feature','?')}: {r.get('value','?')} (contrib {r.get('contribution','?')})")
        else:
            st.write(f"- {r}")
    if st.checkbox("Show full JSON response"):
        st.json(res)

# --- Single sample prediction flow ---
if submitted:
    payload = {
        "sensor_id": "streamlit_demo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lat": lat or None,
        "lon": lon or None,
        "readings": {"pH": ph, "tds": tds, "turbidity": turbidity, "temp": temp},
        "user_notes": notes
    }

    if use_remote and remote_url:
        st.info("Calling remote API...")
        res = call_remote_api(payload)
        show_result(res, debug_payload=payload)
    else:
        st.info("Using local prediction logic (demo thresholds).")
        res = local_predict_row(payload["readings"])
        show_result(res, debug_payload=payload)

    # optional LLM explanation
    if use_llm:
        if not openai_key:
            st.warning("LLM selected but OPENAI API KEY not provided.")
        else:
            st.info("Generating natural language explanation (LLM)...")
            try:
                import openai
                openai.api_key = openai_key
                context = {
                    "readings": payload["readings"],
                    "prediction": res.get("prediction"),
                    "risk_score": res.get("risk_score"),
                    "reasons": res.get("reasons")
                }
                prompt = (
                    "You are WaterGuard. Summarize the following sensor readings and explain "
                    "why the water is safe or unsafe, give 3 actionable steps and one sentence "
                    "about how confident the model is.\n\n"
                    f"{context}"
                )
                completion = openai.ChatCompletion.create(
                    model="gpt-4o-mini",  # replace with your allowed model
                    messages=[{"role":"system","content":"You are WaterGuard."},
                              {"role":"user","content":prompt}],
                    max_tokens=300,
                )
                explanation = completion["choices"][0]["message"]["content"]
                st.markdown("**LLM Explanation:**")
                st.write(explanation)
            except Exception as e:
                st.error("LLM call failed: " + str(e))

# --- Batch prediction display ---
if "batch_df" in st.session_state:
    df = st.session_state["batch_df"]
    st.header("Batch results (preview)")
    results = []
    for _, row in df.iterrows():
        r = local_predict_row(row)
        results.append({**row.to_dict(), **r})
    out = pd.DataFrame(results)
    st.dataframe(out.head(50))
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")