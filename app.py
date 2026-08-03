import streamlit as st
import joblib
import pandas as pd

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Property Price Predictor")


# ==============================
# Load Model
# ==============================
try:
    model = joblib.load("model.pkl")
    st.success("✅ Model loaded successfully")

    # Dynamic model name
    model_name = type(model).__name__
    st.info(f"🤖 Model Used: {model_name}")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()


# ==============================
# Load Metrics
# ==============================
try:
    metrics = joblib.load("metrics.pkl")

    st.sidebar.header("📊 Model Performance")

    st.sidebar.metric("R² Score", f"{metrics['r2']:.2f}")
    st.sidebar.metric("MAE", f"₹ {metrics['mae']:,.0f}")
    st.sidebar.metric("RMSE", f"₹ {metrics['rmse']:,.0f}")

except Exception:
    st.sidebar.warning("Metrics file not available")


# ==============================
# Input Section
# ==============================
st.markdown("### 📋 Enter Property Details")

area = st.number_input(
    "📏 Area (sq ft)",
    min_value=500,
    value=1000
)

bedrooms = st.number_input(
    "🛏 Bedrooms",
    min_value=1,
    value=2
)

location = st.slider(
    "📍 Location Score",
    1,
    5,
    3
)


# ==============================
# Prediction Section
# ==============================
if st.button("🚀 Predict Price"):

    if area <= 0 or bedrooms <= 0:
        st.error("⚠️ Enter valid inputs")

    else:
        input_data = pd.DataFrame(
            [[area, bedrooms, location]],
            columns=["area", "bedrooms", "location"]
        )

        with st.spinner("Predicting..."):
            prediction = model.predict(input_data)

        st.success(
            f"💰 Estimated Price: ₹ {prediction[0]:,.0f}"
        )


# ==============================
# Feature Importance
# ==============================
if hasattr(model, "feature_importances_"):

    st.markdown("---")
    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame({
        "Feature": ["Area", "Bedrooms", "Location"],
        "Importance": model.feature_importances_
    })

    st.bar_chart(
        importance.set_index("Feature")
    )


# ==============================
# Footer
# ==============================
st.markdown("---")
st.caption("Built with Python + Machine Learning + Streamlit")
