import streamlit as st
import joblib
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Property Price Predictor")


# ---------------------------
# LOAD MODEL
# ---------------------------
try:
    model = joblib.load("model.pkl")

    st.success("✅ Model loaded successfully")
    st.info(f"🤖 Model Used: {type(model).__name__}")

except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()


# ---------------------------
# USER INPUT
# ---------------------------
st.markdown("### 📋 Enter Property Details")


area = st.number_input(
    "📏 Area (sq ft)",
    min_value=0,
    value=0,
    step=100
)


bedrooms = st.number_input(
    "🛏 Number of Rooms",
    min_value=0,
    value=0,
    step=1
)


location = st.slider(
    "📍 Location Score",
    min_value=1,
    max_value=5,
    value=3
)


# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🚀 Predict Price"):

    # Validation
    if area <= 0:
        st.error("⚠️ Area must be greater than 0")

    elif bedrooms <= 0:
        st.error("⚠️ Number of rooms must be greater than 0")

    else:

        try:

            # Get trained model feature names
            input_data = pd.DataFrame(
                [[area, bedrooms, location]],
                columns=model.feature_names_in_
            )


            prediction = model.predict(input_data)[0]


            if prediction <= 0:
                st.warning("⚠️ Invalid prediction generated")

            else:
                st.success(
                    f"💰 Estimated Property Price: ₹ {prediction:,.2f}"
                )


            # ---------------------------
            # FEATURE IMPORTANCE GRAPH
            # ---------------------------
            if hasattr(model, "feature_importances_"):

                st.markdown("---")
                st.subheader("📊 Feature Importance")


                feature_data = pd.DataFrame(
                    {
                        "Feature": model.feature_names_in_,
                        "Importance": model.feature_importances_
                    }
                )


                st.bar_chart(
                    feature_data.set_index("Feature")
                )


        except Exception as e:
            st.error(f"❌ Prediction Error: {e}")


# Footer
st.markdown("---")
st.caption("Built with Python + Machine Learning + Streamlit 🚀")
