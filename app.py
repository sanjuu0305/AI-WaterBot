import streamlit as st
from streamlit_lottie import st_lottie
import json
from geopy.geocoders import Nominatim
import requests
from datetime import datetime, timedelta
import google.generativeai as genai
import plotly.graph_objects as go
import pandas as pd

# ----------------- Gemini API Key Setup -----------------
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # 🔒 Replace with your key
model = genai.GenerativeModel("models/gemini-1.5-flash")

# ----------------- Load Animation -----------------
def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

hello_animation = load_lottie_file("hello_animation.json")

# ----------------- Utility Functions -----------------
def get_coordinates(location_name):
    geolocator = Nominatim(user_agent="water-advisor")
    location = geolocator.geocode(location_name)
    if location:
        return location.latitude, location.longitude
    return None, None

def get_weather_data(lat, lon):
    past_end = datetime.utcnow().date() - timedelta(days=1)
    past_start = past_end - timedelta(days=9)
    future_start = datetime.utcnow().date()
    url_past = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={past_start}&end_date={past_end}"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        f"&timezone=auto"
    )
    url_future = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=precipitation_sum,temperature_2m_max,temperature_2m_min"
        f"&forecast_days=10&timezone=auto"
    )
    return requests.get(url_past).json().get("daily", {}), requests.get(url_future).json().get("daily", {})

def safe_avg(values):
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else 0

def safe_sum(values):
    return sum([v for v in values if v is not None])

def get_advice(location, past, future, user_query, language):
    prompt = f"""
You are a Clean Water and Sanitation AI Advisor helping a person in {location}.
Respond in {language}. Be short, clear, and friendly.

**Past 10 Days:**
- Rain: {safe_sum(past.get('precipitation_sum', [])):.1f} mm
- Avg Max Temp: {safe_avg(past.get('temperature_2m_max', [])):.1f}°C
- Avg Min Temp: {safe_avg(past.get('temperature_2m_min', [])):.1f}°C

**Next 10 Days (Forecast):**
- Rain: {safe_sum(future.get('precipitation_sum', [])):.1f} mm
- Avg Max Temp: {safe_avg(future.get('temperature_2m_max', [])):.1f}°C
- Avg Min Temp: {safe_avg(future.get('temperature_2m_min', [])):.1f}°C

User Question: {user_query}
Give suggestions about water safety, flood, hygiene, sanitation, or clean drinking water.
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="WaterBot", layout="centered")
st.title("🚰 WaterBot - AI for Clean Water & Sanitation")

st.markdown("Check water safety, sanitation status, and hygiene tips using AI and weather data.")

if hello_animation:
    st_lottie(hello_animation, height=200)

# -------- Input Location --------
location = st.text_input("📍 Enter your area (village/city/state):")

if location:
    with st.spinner("🌐 Fetching location data..."):
        lat, lon = get_coordinates(location)
        if not lat:
            st.error("❌ Location not found. Try again.")
        else:
            st.success(f"📍 Coordinates: ({lat}, {lon})")
            past, future = get_weather_data(lat, lon)

            # Past Weather
            df_past = pd.DataFrame(past)
            df_future = pd.DataFrame(future)
            df_past["date"] = pd.date_range(end=datetime.today() - timedelta(days=1), periods=10)
            df_future["date"] = pd.date_range(start=datetime.today(), periods=10)

            st.markdown("### 🌧️ Weather Overview (Past & Future)")
            fig = go.Figure()

            fig.add_trace(go.Scatter(x=df_past["date"], y=df_past["temperature_2m_max"],
                                     name="Past Max Temp", mode="lines+markers"))
            fig.add_trace(go.Bar(x=df_past["date"], y=df_past["precipitation_sum"],
                                 name="Past Rain", marker_color='blue'))

            fig.add_trace(go.Scatter(x=df_future["date"], y=df_future["temperature_2m_max"],
                                     name="Forecast Max Temp", mode="lines+markers", line=dict(dash='dash')))
            fig.add_trace(go.Bar(x=df_future["date"], y=df_future["precipitation_sum"],
                                 name="Forecast Rain", marker_color='lightblue'))

            fig.update_layout(title="📊 Rainfall & Temperature Trends",
                              xaxis_title="Date", yaxis_title="Temp (°C) / Rain (mm)",
                              legend_title="Data", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # User Query & Advice
            query = st.text_input("💬 Ask a question about clean water, hygiene, flood risk, or sanitation:")
            language = st.radio("🌐 Choose language:", ["English", "Gujarati", "Hindi"], horizontal=True)

            if query:
                with st.spinner("🤖 Generating advice..."):
                    answer = get_advice(location, past, future, query, language)
                    st.markdown("### ✅ AI Suggestion:")
                    st.success(answer)