# ---------------------------
# INPUT SECTION
# ---------------------------
st.markdown("### 📋 Enter Property Details")

area = st.number_input(
    "📏 Area (sq ft)",
    value=None,
    placeholder="Enter area in sq ft",
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
    min_value=1,
    max_value=5,
    value=3
)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🚀 Predict Price"):

    # Validation
    if area is None or bedrooms is None:
        st.error("⚠️ Please enter Area and Bedrooms")
        
    elif area <= 0 or bedrooms <= 0:
        st.error("⚠️ Area and Bedrooms must be greater than 0")
        
    else:
        try:
            input_data = pd.DataFrame(
                [[area, bedrooms, location]],
                columns=["area", "bedrooms", "location"]
            )

            prediction = model.predict(input_data)[0]

            if prediction <= 0:
                st.warning("⚠️ Invalid prediction value generated")
            else:
                st.success(
                    f"💰 Estimated Price: ₹ {prediction:,.2f}"
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")
