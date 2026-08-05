import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# Debug Info (VERY IMPORTANT)
# -------------------------------
st.write("🔍 Debug Info")
st.write("Current files:", os.listdir())

# -------------------------------
# Load Model
# -------------------------------
model = None

if os.path.exists("model.pkl"):
    try:
        model = joblib.load("model.pkl")
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error("❌ model.pkl NOT FOUND in repo")

# Stop app if model not loaded
if model is None:
    st.stop()

# -------------------------------
# UI STARTS HERE
# -------------------------------
st.title("🏠 Property Price Predictor")

st.markdown("### Enter Property Details")

# Inputs
area = st.number_input("Area (sq ft)", 500, 10000, 1000)
bedrooms = st.number_input("Bedrooms", 1, 10, 2)
location = st.selectbox("Location", ["Low", "Medium", "High"])

# Mapping
location_map = {"Low": 1, "Medium": 2, "High": 3}
location_score = location_map[location]

# Predict
if st.button("Predict Price"):
    try:
        features = np.array([[area, bedrooms, location_score]])
        prediction = model.predict(features)

        st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f} Lakhs")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
