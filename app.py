import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Property Price Predictor")


# Load model
try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Input Section
st.markdown("### 📋 Enter Property Details")

area = st.number_input(
    "📏 Area (sq ft)",
    value=None,
    placeholder="Enter area",
    step=100
)

bedrooms = st.number_input(
    "🛏 Bedrooms",
    value=None,
    placeholder="Enter number of bedrooms",
    step=1
)

location = st.slider(
    "📍 Location Score",
    1,
    5,
    3
)


# Prediction
if st.button("🚀 Predict Price"):

    if area is None or bedrooms is None:
        st.error("⚠️ Please enter Area and Bedrooms")

    elif area <= 0 or bedrooms <= 0:
        st.error("⚠️ Values must be greater than zero")

    else:
        input_data = pd.DataFrame(
            [[area, bedrooms, location]],
            columns=["area", "bedrooms", "location"]
        )

        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 Estimated Price: ₹ {prediction:,.2f}"
        )
