import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("🏠 Bangalore House Price Predictor")

area = st.number_input("Area (sq ft)", 500, 5000)
bedrooms = st.number_input("Bedrooms", 1, 6)

location = st.selectbox(
    "Location",
    [
        "Indiranagar", "Whitefield", "HSR Layout",
        "BTM", "Electronic City", "Marathahalli",
        "Hebbal", "Yelahanka", "Kengeri", "Bannerghatta"
    ]
)

if st.button("Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms]], columns=["area", "bedrooms"])

    # Add all location columns
    for col in columns:
        if col.startswith("location_"):
            input_data[col] = 0

    # Set selected location = 1
    input_data[f"location_{location}"] = 1

    # Reorder columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Price: ₹ {round(prediction,2)} Lakhs")
