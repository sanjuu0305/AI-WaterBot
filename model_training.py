# snippet: display readings on a map in Streamlit
import streamlit as st
import pandas as pd
import pydeck as pdk

st.title("Local Water Readings Map")

# Example dataframe (replace with uploaded CSV / DB)
df = pd.DataFrame([
    {"site":"well_01","lat":21.1702,"lon":72.8311,"pH":6.2,"tds":820,"turbidity":15,"bacteria":"positive"},
    {"site":"tap_02","lat":21.1715,"lon":72.8330,"pH":7.1,"tds":120,"turbidity":1,"bacteria":"negative"},
    {"site":"pond_03","lat":21.1690,"lon":72.8290,"pH":8.9,"tds":400,"turbidity":8,"bacteria":"negative"},
])

# Define simple risk scoring
def risk_from_row(r):
    score = 0
    if r['pH'] < 6.5 or r['pH'] > 8.5: score += 1
    if r['tds'] > 500: score += 2
    elif r['tds'] > 300: score += 1
    if r['turbidity'] > 5: score += 1
    if r.get('bacteria') == 'positive': score += 3
    if score >= 3: return "unsafe"
    if score == 1 or score == 2: return "caution"
    return "safe"

df['risk'] = df.apply(risk_from_row, axis=1)
color_map = {"safe":[0,200,0],"caution":[255,165,0],"unsafe":[200,0,0]}
df['color'] = df['risk'].map(color_map)

# Pydeck map
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position='[lon, lat]',
    get_fill_color='color',
    get_radius=50,
    pickable=True
)
view_state = pdk.ViewState(latitude=df['lat'].mean(), longitude=df['lon'].mean(), zoom=13)
r = pdk.Deck(layers=[layer], initial_view_state=view_state)
st.pydeck_chart(r)

# show table and flagged sites
st.subheader("Flagged sites (unsafe / caution)")
st.dataframe(df.sort_values(by='risk', ascending=False))