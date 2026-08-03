import streamlit as st
import pandas as pd
import joblib


# ---------------------------
# PAGE CONFIG
# ---------------------------

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

    model = joblib.load(
        "model.pkl"
    )

    st.success(
        "✅ Model loaded successfully"
    )

except Exception as e:

    st.error(
        f"❌ Model loading failed: {e}"
    )

    st.stop()



# ---------------------------
# MODEL DETAILS
# ---------------------------

st.sidebar.header(
    "🤖 Model Information"
)


st.sidebar.write(
    f"Model: {type(model).__name__}"
)


st.sidebar.write(
    "Features:"
)


st.sidebar.write(
    list(model.feature_names_in_)
)



# ---------------------------
# INPUT SECTION
# ---------------------------

st.header(
    "🏡 Enter Property Details"
)



col1, col2 = st.columns(2)



with col1:


    area = st.number_input(
        "📏 Area (sq ft)",
        min_value=0,
        value=0,
        step=100
    )


    bedrooms = st.number_input(
        "🛏 Bedrooms",
        min_value=0,
        value=0,
        step=1
    )


    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=0,
        value=0,
        step=1
    )



with col2:


    age = st.number_input(
        "🏚 Property Age (Years)",
        min_value=0,
        value=0,
        step=1
    )


    parking = st.selectbox(
        "🚗 Parking Available",
        [
            "No",
            "Yes"
        ]
    )



property_type = st.selectbox(
    "🏢 Property Type",
    [
        "Apartment",
        "Villa",
        "Independent House"
    ]
)



location = st.selectbox(
    "📍 Location",
    [
        "Electronic City",
        "Whitefield",
        "HSR Layout",
        "Marathahalli",
        "Koramangala"
    ]
)



# ---------------------------
# ENCODING
# ---------------------------

property_mapping = {

    "Apartment": 0,
    "Villa": 1,
    "Independent House": 2

}


location_mapping = {

    "Electronic City": 0,
    "Whitefield": 1,
    "HSR Layout": 2,
    "Marathahalli": 3,
    "Koramangala": 4

}



# ---------------------------
# PREDICTION
# ---------------------------

if st.button(
    "🚀 Predict Price"
):


    # Validation

    if area <= 0:

        st.error(
            "⚠️ Area must be greater than 0"
        )


    elif bedrooms <= 0:

        st.error(
            "⚠️ Bedrooms must be greater than 0"
        )


    elif bathrooms <= 0:

        st.error(
            "⚠️ Bathrooms must be greater than 0"
        )


    else:


        try:


            input_data = pd.DataFrame(

                {

                    "area":[area],

                    "bedrooms":[bedrooms],

                    "bathrooms":[bathrooms],

                    "parking":[
                        1 if parking=="Yes" else 0
                    ],

                    "age":[age],

                    "property_type":[
                        property_mapping[property_type]
                    ],

                    "location":[
                        location_mapping[location]
                    ]

                }

            )



            # Match model training columns

            input_data = input_data[
                model.feature_names_in_
            ]



            prediction = model.predict(
                input_data
            )[0]



            if prediction <= 0:

                st.warning(
                    "⚠️ Invalid prediction generated"
                )


            else:

                st.success(
                    f"💰 Estimated Price: ₹ {prediction:,.0f}"
                )



                # Price Range

                lower = prediction * 0.90

                upper = prediction * 1.10


                st.info(
                    f"""
📊 Expected Price Range

Minimum:
₹ {lower:,.0f}


Maximum:
₹ {upper:,.0f}
"""
                )



                # Property Summary

                st.subheader(
                    "🏠 Property Summary"
                )


                summary = pd.DataFrame(
                    {

                        "Feature":[

                            "Area",

                            "Bedrooms",

                            "Bathrooms",

                            "Parking",

                            "Age",

                            "Property Type",

                            "Location"

                        ],


                        "Value":[

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


                st.table(
                    summary
                )



        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )



# ---------------------------
# FEATURE IMPORTANCE GRAPH
# ---------------------------


st.markdown("---")


st.subheader(
    "📈 Feature Importance"
)



if hasattr(
    model,
    "feature_importances_"
):


    importance = pd.DataFrame(

        {

            "Feature":
            model.feature_names_in_,


            "Importance":
            model.feature_importances_

        }

    )


    st.bar_chart(
        importance.set_index(
            "Feature"
        )
    )


else:

    st.info(
        "Feature importance not available"
    )



# ---------------------------
# FOOTER
# ---------------------------

st.markdown("---")

st.caption(
    "Built using Python + Machine Learning + Streamlit 🚀"
)
