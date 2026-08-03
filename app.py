import streamlit as st
import joblib
import pandas as pd

# Page configuration (must be first Streamlit command)
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

    # Dynamic model name
    model_name = type(model).__name__
    st.info(f"🤖 Model Used: {model_name}")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()


# Load metrics
try:
    metrics = joblib.load("metrics.pkl")

    
    st.sidebar.header("📊 Model Performance")

    st.sidebar.write(
        f"R² Score: {metrics['r2']:.2f}"
    )

    st.sidebar.write(
        f"MAE: ₹ {metrics['mae']:,.0f}"
    )

    st.sidebar.write(
        f"RMSE: ₹ {metrics['rmse']:,.0f}"
    )

except Exception:
    st.sidebar.warning("Metrics file not available")


# Input section
st.markdown("### 📋 Enter Property Details")


area = st.number_input(
    "📏 Area (sq ft)",
    min_value=0
)

bedrooms = st.number_input(
    "🛏 Bedrooms",
    min_value=0
)

location = st.slider(
    "📍 Location Score",
    1,
    5
)


# Prediction
if st.button("🚀 Predict Price"):

    if area <= 0 or bedrooms <= 0:
        st.error("⚠️ Enter valid inputs")

    else:

       input_data = pd.DataFrame(
    [[area, bedrooms, location]],
    columns=["area", "bedrooms", "location"]
)

prediction = model.predict(input_data)

        st.success(
            f"💰 Estimated Price: ₹ {prediction[0]:,.2f}"
        )


# Feature importance
if hasattr(model, "feature_importances_"):

    st.markdown("---")
    st.subheader("📈 Feature Importance")

    importance = pd.DataFrame(
        {
            "Feature": [
                "Area",
                "Bedrooms",
                "Location"
            ],
            "Importance": model.feature_importances_
        }
    )

    st.bar_chart(
        importance.set_index("Feature")
    )


st.markdown("---")
st.caption("Built with Python + Machine Learning + Streamlit")
