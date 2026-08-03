import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠"
)


st.title(
    "🏠 Property Price Predictor"
)



# Load files

model=joblib.load(
    "model.pkl"
)


property_encoder=joblib.load(
    "property_encoder.pkl"
)


location_encoder=joblib.load(
    "location_encoder.pkl"
)



# ---------------------------
# INPUTS
# ---------------------------

st.header(
    "📋 Property Details"
)



col1,col2=st.columns(2)



with col1:

    area=st.number_input(
        "Area (sq ft)",
        min_value=0
    )


    bedrooms=st.number_input(
        "Bedrooms",
        min_value=0
    )


    bathrooms=st.number_input(
        "Bathrooms",
        min_value=0
    )



with col2:

    age=st.number_input(
        "Property Age",
        min_value=0
    )


    parking=st.selectbox(
        "Parking Available",
        ["No","Yes"]
    )



property_type=st.selectbox(
    "Property Type",
    [
        "Apartment",
        "Villa",
        "Independent House"
    ]
)



location=st.selectbox(
    "Location",
    [
        "Electronic City",
        "Whitefield",
        "HSR Layout",
        "Marathahalli",
        "Koramangala"
    ]
)



# ---------------------------
# PREDICTION
# ---------------------------


if st.button(
    "🚀 Predict Price"
):


    if area<=0 or bedrooms<=0:

        st.error(
            "Enter valid Area and Bedrooms"
        )


    else:


        input_df=pd.DataFrame(

        [[
            area,
            bedrooms,
            bathrooms,
            1 if parking=="Yes" else 0,
            age,

            property_encoder.transform(
                [property_type]
            )[0],

            location_encoder.transform(
                [location]
            )[0]

        ]],

        columns=model.feature_names_in_

        )



        result=model.predict(
            input_df
        )[0]



        st.success(
            f"💰 Estimated Price ₹ {result:,.0f}"
        )



        # Price range

        st.info(
            f"""
Expected Range:

₹ {result*0.9:,.0f}

-
₹ {result*1.1:,.0f}
"""
        )



# ---------------------------
# GRAPHS
# ---------------------------


st.subheader(
    "📊 Feature Importance"
)


importance=pd.DataFrame({

"Feature":
model.feature_names_in_,

"Importance":
model.feature_importances_

})


st.bar_chart(
    importance.set_index(
        "Feature"
    )
)



st.subheader(
    "📈 Feature Contribution %"
)


importance["Percentage"]=(
importance["Importance"]*100
)


st.line_chart(
importance.set_index(
    "Feature"
)[
"Percentage"
]
)
