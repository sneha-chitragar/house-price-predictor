import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Model Safely
# -------------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -------------------------------
# App Title
# -------------------------------
st.title("🏠 Property Price Predictor")

st.markdown("### Enter Property Details")

# -------------------------------
# User Inputs
# -------------------------------
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)

location = st.selectbox("Location Type", ["Low", "Medium", "High"])

# Convert categorical to numeric
location_map = {"Low": 1, "Medium": 2, "High": 3}
location_score = location_map[location]

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Price"):

    try:
        # Ensure feature order matches training
        features = np.array([[area, bedrooms, location_score]])

        prediction = model.predict(features)

        st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f} Lakhs")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -------------------------------
# Extra Info
# -------------------------------
st.markdown("---")
st.markdown("### 📊 Model Info")
st.write("Model Used: Random Forest Regressor")
st.write("Features: Area, Bedrooms, Location")
