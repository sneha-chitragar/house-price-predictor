import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Predictor")

# -----------------------
# CHECK FILES EXIST
# -----------------------
if not os.path.exists("model.pkl"):
    st.error("❌ model.pkl not found. Upload it to your repo.")
    st.stop()

# -----------------------
# LOAD MODEL SAFELY
# -----------------------
try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Model failed to load: {e}")
    st.stop()

# -----------------------
# INPUTS
# -----------------------
st.subheader("Enter Details")

area = st.number_input("Area (sq ft)", min_value=1, step=50)
bedrooms = st.number_input("Bedrooms", min_value=1, step=1)
location = st.slider("Location Score", 1, 5, 3)

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict Price"):

    # Validation
    if area <= 0 or bedrooms <= 0:
        st.error("❌ Values must be greater than 0")
        st.stop()

    try:
        # Try with column names
        input_df = pd.DataFrame(
            [[area, bedrooms, location]],
            columns=["area", "bedrooms", "location"]
        )

        prediction = model.predict(input_df)[0]

    except Exception:
        # fallback (if model doesn't use feature names)
        input_df = [[area, bedrooms, location]]
        prediction = model.predict(input_df)[0]

    if prediction <= 0:
        st.warning("⚠️ Invalid prediction. Try different values.")
    else:
        st.success(f"💰 Estimated Price: ₹ {prediction:,.2f}")
