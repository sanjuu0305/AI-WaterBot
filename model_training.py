# api.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class WaterData(BaseModel):
    pH: float
    tds: float
    turbidity: float
    temp: float

@app.post("/predict")
def predict(data: WaterData):
    ph, tds, turb, temp = data.pH, data.tds, data.turbidity, data.temp
    score = 0
    if ph < 6.5 or ph > 8.5: score += 0.3
    if tds > 500: score += 0.4
    if turb > 5: score += 0.2
    if temp > 35: score += 0.1
    return {
        "prediction": "unsafe" if score > 0.5 else "safe",
        "risk_score": score,
        "reasons": [],
        "action": "Boil/filter before use" if score > 0.5 else "Water safe"
    }