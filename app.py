"""Standalone Streamlit app for UFO sighting country prediction."""
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("models/ufo-model.pkl", "rb"))
countries = ["Australia", "Canada", "Germany", "UK", "US"]

st.title("UFO Sighting Predictor")
st.write("Predict which country a UFO sighting is likely from based on duration, latitude, and longitude.")

seconds = st.number_input("Duration (seconds)", min_value=1, max_value=60, value=30)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.0, format="%.4f")
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-100.0, format="%.4f")

if st.button("Predict"):
    features = np.array([[seconds, latitude, longitude]])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]

    st.success(f"Likely country: **{countries[prediction[0]]}**")

    st.write("### Probability Distribution")
    prob_dict = {c: float(p) for c, p in zip(countries, probabilities)}
    st.bar_chart(prob_dict)
