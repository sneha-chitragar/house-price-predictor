import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

st.title("🏠 Property Price Predictor")

st.write("Enter property details:")

# Inputs
area = st.number_input("Area (sq ft)", min_value=500, max_value=5000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
location = st.selectbox("Location", ["Low", "Medium", "High"])

# Convert location to numeric
location_map = {"Low": 1, "Medium": 2, "High": 3}
location_score = location_map[location]

# Prediction
if st.button("Predict Price"):
    features = np.array([[area, bedrooms, location_score]])
    prediction = model.predict(features)

    st.success(f"Estimated Price: ₹ {prediction[0]:.2f} Lakhs")
