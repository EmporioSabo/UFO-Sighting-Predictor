"""FastAPI backend for UFO sighting country prediction."""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("models/ufo-model.pkl", "rb"))
countries = ["Australia", "Canada", "Germany", "UK", "US"]


class UFOInput(BaseModel):
    seconds: float
    latitude: float
    longitude: float


@app.get("/")
def index():
    return {"ok": True}


@app.post("/predict/")
async def predict(data: UFOInput):
    features = np.array([[data.seconds, data.latitude, data.longitude]])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    return {
        "prediction": countries[prediction[0]],
        "probabilities": dict(zip(countries, probabilities.tolist()))
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
