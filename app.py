import streamlit as st
import pandas as pd
import joblib


# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)


st.title("🏠 Property Price Predictor")


# ---------------------------------
# LOAD MODEL
# ---------------------------------

try:

    model = joblib.load("model.pkl")

    st.success("✅ Model loaded successfully")


except Exception as e:

    st.error(
        f"❌ Model loading failed: {e}"
    )

    st.stop()



# ---------------------------------
# MODEL INFORMATION
# ---------------------------------

st.sidebar.header(
    "🤖 Model Information"
)


st.sidebar.write(
    "Model:"
)

st.sidebar.write(
    type(model).__name__
)


st.sidebar.write(
    "Features:"
)


st.sidebar.write(
    list(model.feature_names_in_)
)



# ---------------------------------
# USER INPUT
# ---------------------------------

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



# ---------------------------------
# ENCODING
# ---------------------------------

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



# ---------------------------------
# PREDICTION
# ---------------------------------

if st.button(
    "🚀 Predict Price"
):


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


            # Match model columns

            input_data = input_data[
                model.feature_names_in_
            ]


            prediction = model.predict(
                input_data
            )[0]



            if prediction <= 0:

                st.warning(
                    "⚠️ Invalid prediction"
                )


            else:


                st.success(
                    f"💰 Estimated Property Price: ₹ {prediction:,.0f}"
                )


                # Price Range

                st.info(

                    f"""
📊 Expected Market Range

Minimum:
₹ {prediction*0.90:,.0f}


Maximum:
₹ {prediction*1.10:,.0f}
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


                st.table(summary)



        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )



# ---------------------------------
# GRAPH 1 : FEATURE IMPORTANCE
# ---------------------------------

st.markdown("---")


st.subheader(
    "📈 Feature Importance"
)



if hasattr(
    model,
    "feature_importances_"
):


    feature_data = pd.DataFrame(

        {

            "Feature":
            model.feature_names_in_,


            "Importance":
            model.feature_importances_

        }

    )


    st.bar_chart(

        feature_data.set_index(
            "Feature"
        )

    )



# ---------------------------------
# GRAPH 2 : FEATURE CONTRIBUTION %
# ---------------------------------

st.subheader(
    "📊 Feature Contribution Percentage"
)



if hasattr(
    model,
    "feature_importances_"
):


    contribution = pd.DataFrame(

        {

            "Feature":
            model.feature_names_in_,


            "Percentage":
            model.feature_importances_ * 100

        }

    )


    st.line_chart(

        contribution.set_index(
            "Feature"
        )

    )



# ---------------------------------
# FOOTER
# ---------------------------------

st.markdown("---")


st.caption(
    "Built with Python + Machine Learning + Streamlit 🚀"
)
