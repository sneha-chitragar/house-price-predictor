import streamlit as st
import joblib
import numpy as np

st.title("App is running 🚀")

st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="centered")

# Load model safely
try:
    model = joblib.load('model.pkl')
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")

# UI
st.markdown("## 🏠 Property Price Predictor")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
location = st.slider("Location", 1, 5)

if st.button("Predict"):
    if area <= 0 or bedrooms <= 0:
        st.error("Enter valid inputs")
    else:
        prediction = model.predict([[area, bedrooms, location]])
        st.success(f"Price: ₹ {prediction[0]:,.2f}")
