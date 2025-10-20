# --- AREA-BASED ASSESSMENT USING GENERATIVE AI ---
import streamlit as st
import requests
import time
import json
import math

# Requires openai pip package if use_llm True
try:
    import openai
except Exception:
    openai = None

st.markdown("---")
st.header("🌍 Area-level water safety suggestion (Generative AI)")

area_name = st.text_input("Enter area / locality name (city, neighborhood, village)", value="", help="e.g. 'Paldi, Ahmedabad' or 'Sector 21, Gandhinagar'")
use_geocode = st.checkbox("Try to geocode area name (find nearby uploaded readings)", value=True)
assess_btn = st.button("Assess area now")

# Helper: simple Haversine distance (km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

# Helper: geocode using Nominatim (OpenStreetMap) - may rate-limit; considered optional
def geocode_area(area):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": area, "format": "json", "limit": 1}
        r = requests.get(url, params=params, headers={"User-Agent":"CleanWaterApp/1.0 (+your_email@example.com)"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", "")
        return None
    except Exception as e:
        st.warning(f"Geocoding failed: {e}")
        return None

# Helper: collect nearby readings from session (if user uploaded a CSV earlier)
def nearby_readings(lat, lon, radius_km=5.0):
    # expects st.session_state["all_readings"] as a DataFrame with 'lat' and 'lon' columns
    import pandas as pd
    df = st.session_state.get("all_readings")
    if df is None or not hasattr(df, "iterrows"):
        return []
    rows = []
    for _, r in df.iterrows():
        try:
            rlat = float(r.get("lat") or r.get("latitude") or r.get("Latitude") or 0)
            rlon = float(r.get("lon") or r.get("longitude") or r.get("Longitude") or 0)
        except Exception:
            continue
        d = haversine(lat, lon, rlat, rlon)
        if d <= radius_km:
            row = r.to_dict()
            row["_distance_km"] = round(d, 3)
            rows.append(row)
    # sort by distance
    rows = sorted(rows, key=lambda x: x["_distance_km"])
    return rows

# Build LLM prompt securely and conservatively
def build_area_prompt(area_name, geocode_info=None, nearby=None):
    parts = []
    parts.append("You are WaterGuard, an assistant that gives evidence-aware, cautious suggestions about water safety for a given locality.")
    parts.append("Always: 1) say if water is likely 'safe', 'unsafe', or 'uncertain'. 2) give a numeric confidence (0-1). 3) list the main reasons. 4) provide 3 actionable next steps. 5) explicitly state uncertainty sources and recommend lab confirmation when appropriate.")
    if geocode_info:
        parts.append(f"Area name: {area_name}. Geocoded location: {geocode_info[2]} (lat={geocode_info[0]}, lon={geocode_info[1]}).")
    else:
        parts.append(f"Area name: {area_name}. No reliable geocode given.")
    if nearby:
        parts.append("Nearby sensor readings (most recent first). Each row: site, lat, lon, pH, tds, turbidity, temp, bacteria (if available), timestamp, distance_km.")
        for r in nearby[:8]:
            # add compact reading lines
            parts.append(json.dumps({
                "site": r.get("site") or r.get("site_id") or r.get("siteName"),
                "lat": r.get("lat"),
                "lon": r.get("lon"),
                "pH": r.get("pH"),
                "tds": r.get("tds"),
                "turbidity": r.get("turbidity"),
                "temp": r.get("temp"),
                "bacteria": r.get("bacteria"),
                "timestamp": r.get("timestamp"),
                "distance_km": r.get("_distance_km")
            }))
    else:
        parts.append("No nearby sensor readings available in the app dataset.")

    # safety thresholds to anchor reasoning (so LLM uses them)
    parts.append("Use these example thresholds to judge chemical/microbial risk: pH safe range 6.5-8.5; TDS: <300 good, 300-500 moderate, >500 high; Turbidity: <1 ideal, >5 concerning; any positive E.coli/coliform is immediate flag. But emphasize uncertainty and recommend lab for microbes.")
    parts.append("Output must be a single JSON object only, with keys: prediction ('safe'|'unsafe'|'uncertain'), confidence (0-1), reasons (list), recommended_action (list), notes_about_uncertainty (string). Keep reasons short (1-2 sentences each).")

    return "\n\n".join(parts)

# Call the LLM (OpenAI)
def call_llm_assess(prompt, openai_key, model="gpt-4o-mini"):
    if openai is None:
        return {"error":"openai package not installed"}
    try:
        openai.api_key = openai_key
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role":"system","content":"You are WaterGuard, an assistant that gives cautious, evidence-aware water safety guidance."},
                {"role":"user","content":prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        text = completion["choices"][0]["message"]["content"]
        return {"text": text}
    except Exception as e:
        return {"error": str(e)}

# Action when user clicks Assess
if assess_btn:
    if not area_name.strip():
        st.warning("Please enter an area name.")
    else:
        geocode_info = None
        nearby = []
        if use_geocode:
            geocode_info = geocode_area(area_name)
            if geocode_info:
                st.success(f"Geocoded: {geocode_info[2]} (lat={geocode_info[0]}, lon={geocode_info[1]})")
                nearby = nearby_readings(geocode_info[0], geocode_info[1], radius_km=5.0)
                if nearby:
                    st.info(f"Found {len(nearby)} nearby uploaded readings within 5 km; including them in assessment.")
                else:
                    st.info("No nearby uploaded readings found within 5 km.")
            else:
                st.info("Could not geocode area; proceeding without nearby readings.")
        else:
            st.info("Geocoding disabled; proceeding without nearby readings.")

        # Build prompt
        prompt = build_area_prompt(area_name, geocode_info=geocode_info, nearby=nearby)

        if not st.session_state.get("OPENAI_API_KEY"):
            # check Streamlit secrets / input
            openai_key = st.secrets.get("OPENAI_API_KEY") if "OPENAI_API_KEY" in st.secrets else st.text_input("OPENAI API KEY (required for LLM)", type="password")
            st.session_state["OPENAI_API_KEY"] = openai_key
        else:
            openai_key = st.session_state["OPENAI_API_KEY"]

        if not openai_key:
            st.error("LLM key not provided. Unable to run generative assessment. You can still upload readings and use local threshold rules.")
        else:
            with st.spinner("Asking the LLM for an evidence-aware suggestion..."):
                llm_res = call_llm_assess(prompt, openai_key=openai_key)
            if "error" in llm_res:
                st.error("LLM call failed: " + llm_res["error"])
            else:
                text = llm_res["text"]
                st.subheader("LLM raw output")
                st.text_area("Raw LLM response (for debugging)", value=text, height=240)

                # Try to parse JSON from LLM output conservatively
                parsed = None
                try:
                    # extract first JSON-looking substring
                    start = text.find("{")
                    end = text.rfind("}")
                    if start != -1 and end != -1:
                        maybe = text[start:end+1]
                        parsed = json.loads(maybe)
                except Exception as e:
                    parsed = None

                if parsed:
                    st.success(f"Prediction: {parsed.get('prediction')}  (confidence {parsed.get('confidence')})")
                    st.markdown("**Reasons:**")
                    for r in parsed.get("reasons", []):
                        st.write("-", r)
                    st.markdown("**Recommended actions:**")
                    for a in parsed.get("recommended_action", []):
                        st.write("-", a)
                    st.markdown("**Uncertainty notes:**")
                    st.write(parsed.get("notes_about_uncertainty", "None provided"))
                else:
                    st.warning("Could not parse JSON cleanly from LLM. See raw output above. The LLM should return a single JSON object — you can adjust the model or prompt if necessary.")