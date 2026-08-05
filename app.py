import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Model
# -------------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------------------
# Application Title
# -------------------------------
st.title("🏠 Property Price Predictor")

st.write(
    "Predict house prices using Machine Learning based on "
    "area, bedrooms, and location."
)

st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Enter Property Details")

area = st.number_input(
    "Area (sq ft)",
    min_value=500,
    max_value=10000,
    value=1000
)

bedrooms = st.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=2
)

location = st.selectbox(
    "Location",
    ["Low", "Medium", "High"]
)

# Convert location to numerical value
location_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

location_score = location_mapping[location]

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price"):

    input_data = np.array(
        [[area, bedrooms, location_score]]
    )

    try:
        prediction = model.predict(input_data)

        st.success(
            f"🏡 Estimated Property Price: ₹ {prediction[0]:,.2f} Lakhs"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# -------------------------------
# Model Information
# -------------------------------
st.markdown("---")

st.subheader("📊 Model Information")

st.write("**Algorithm:** Random Forest Regressor")
st.write("**Input Features:** Area, Bedrooms, Location")
st.write("**Prediction Type:** House Price Estimation")
