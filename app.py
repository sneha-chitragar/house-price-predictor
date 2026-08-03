# ---------------------------
# ENCODING
# ---------------------------

property_mapping = {
    "Apartment": 0,
    "Villa": 1,
    "Independent House": 2
}


location_mapping = {
    "Electronic City": 1,
    "Whitefield": 2,
    "HSR Layout": 3,
    "Marathahalli": 4,
    "Koramangala": 5
}


# ---------------------------
# PREDICTION
# ---------------------------

if st.button("🚀 Predict Price"):

    if area <= 0:
        st.error("⚠️ Area must be greater than 0")

    elif bedrooms <= 0:
        st.error("⚠️ Bedrooms must be greater than 0")

    elif bathrooms <= 0:
        st.error("⚠️ Bathrooms must be greater than 0")

    else:

        try:

            # Create input with SAME feature names as model

            input_data = pd.DataFrame(
                {
                    "area": [area],

                    "bedrooms": [bedrooms],

                    "bathrooms": [bathrooms],

                    "parking": [
                        1 if parking == "Yes" else 0
                    ],

                    "age": [age],

                    "property_type": [
                        property_mapping[property_type]
                    ],

                    "location_score": [
                        location_mapping[location]
                    ]
                }
            )


            # Match model training order

            input_data = input_data[
                model.feature_names_in_
            ]


            prediction = model.predict(
                input_data
            )[0]


            st.success(
                f"💰 Estimated Price: ₹ {prediction:,.0f}"
            )


            # Price range

            st.info(
                f"""
📊 Expected Price Range

Minimum:
₹ {prediction * 0.90:,.0f}

Maximum:
₹ {prediction * 1.10:,.0f}
"""
            )


            # Property Summary

            st.subheader(
                "🏠 Property Summary"
            )


            summary = pd.DataFrame(
                {
                    "Feature": [
                        "Area",
                        "Bedrooms",
                        "Bathrooms",
                        "Parking",
                        "Age",
                        "Property Type",
                        "Location"
                    ],

                    "Value": [
                        area,
                        bedrooms,
                        bathrooms,
                        parking,
                        age,
                        property_type,
                        location
                    ]
                }
            )


            st.table(summary)



        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )
