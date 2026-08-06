import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt



st.set_page_config(

    page_title="Property Price Predictor",

    page_icon="🏠",

    layout="centered"

)



# Load model

model = joblib.load(

    "model_pipeline.pkl"

)



# Load metrics

try:

    with open("metrics.json") as f:

        metrics = json.load(f)

except:

    metrics = {}




st.title(
    "🏠 Property Price Predictor"
)


st.write(
    "Predict property prices using Machine Learning"
)



# Model information

if metrics:

    st.sidebar.header(
        "Model Performance"
    )


    st.sidebar.write(
        "Model:",
        metrics["model"]
    )


    st.sidebar.write(
        "R2 Score:",
        metrics["r2_score"]
    )


    st.sidebar.write(
        "MAE:",
        metrics["mae"]
    )


    st.sidebar.write(
        "RMSE:",
        metrics["rmse"]
    )



st.subheader(
    "Enter Property Details"
)



# Inputs

area = st.number_input(

    "Area (sq ft)",

    min_value=100,

    value=1500

)


bedrooms = st.number_input(

    "Bedrooms",

    min_value=1,

    value=2

)


bathrooms = st.number_input(

    "Bathrooms",

    min_value=1,

    value=2

)


property_age = st.number_input(

    "Property Age",

    min_value=0,

    value=5

)


parking = st.number_input(

    "Parking",

    min_value=0,

    value=1

)



location = st.selectbox(

    "Location",

    [

        "BTM",

        "Indiranagar",

        "Yelahanka"

    ]

)



property_type = st.selectbox(

    "Property Type",

    [

        "Apartment",

        "Villa"

    ]

)



if st.button(
    "Predict Price"
):


    input_data = pd.DataFrame(

        {

        "area":
        [area],


        "bedrooms":
        [bedrooms],


        "bathrooms":
        [bathrooms],


        "property_age":
        [property_age],


        "parking":
        [parking],


        "location":
        [location],


        "property_type":
        [property_type]

        }

    )



    prediction = model.predict(

        input_data

    )



    predicted_price = round(

        prediction[0],

        2

    )



    st.success(

        f"💰 Estimated Price: ₹ {predicted_price} Lakhs"

    )



    # Dynamic Graph

    st.subheader(

        "📊 Property Input Visualization"

    )



    chart_data = pd.DataFrame(

        {

        "Feature":

        [

        "Area",

        "Bedrooms",

        "Bathrooms",

        "Age",

        "Parking"

        ],


        "Value":

        [

        area,

        bedrooms,

        bathrooms,

        property_age,

        parking

        ]

        }

    )



    fig, ax = plt.subplots(

        figsize=(6,3)

    )


    ax.bar(

        chart_data["Feature"],

        chart_data["Value"]

    )


    plt.xticks(

        rotation=45

    )


    plt.tight_layout()



    st.pyplot(fig)



    st.subheader(

        "Selected Input"

    )


    st.dataframe(

        input_data

    )