import streamlit as st
import joblib

# Page config
st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="centered")

# Custom styling
st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #2c3e50;
        font-size: 40px;
        font-weight: bold;
    }
    .subtext {
        text-align: center;
        color: gray;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load('model.pkl')

# Title
st.markdown('<div class="title">🏠 Property Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Predict house prices using Machine Learning</div>', unsafe_allow_html=True)

st.write("")

# Model Info
st.info("🤖 Model Used: Random Forest")

# Input Section
st.markdown("### 📊 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("📏 Area (sq ft)", min_value=0, value=0)
    bedrooms = st.number_input("🛏 Bedrooms", min_value=0, value=0)

with col2:
    location = st.slider("📍 Location Score", 1, 5, value=1)

st.write("")

st.caption("⚠️ Enter realistic values for better prediction accuracy")

# Prediction Button
if st.button("🚀 Predict Price"):

    # ✅ VALIDATION FIX
    if area <= 0 or bedrooms <= 0:
        st.error("⚠️ Please enter valid values (Area > 0 and Bedrooms > 0)")
    
    else:
        prediction = model.predict([[area, bedrooms, location]])
        st.success(f"💰 Estimated Price: ₹ {prediction[0]:,.2f}")
        st.balloons()

# Footer
st.markdown("---")
st.markdown("🔹 Built with Streamlit | Capstone Project")