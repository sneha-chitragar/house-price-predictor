import streamlit as st
import pandas as pd
import joblib
import json
#import matplotlib.pyplot as plt

import plotly.express as px
# Page settings
st.set_page_config(
    page_title="Bangalore Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# Remove number input arrows
st.markdown(
"""
<style>

input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

</style>
""",
unsafe_allow_html=True
)



# Load model

model = joblib.load(
    "model_pipeline.pkl"
)



# Load dataset

df = pd.read_csv(
    "housing_data.csv"
)


df.columns = df.columns.str.strip()



# Load metrics

try:

    with open(
        "metrics.json",
        "r"
    ) as file:

        metrics=json.load(file)

except:

    metrics={}




# Title

st.title(
    "🏠 Bangalore Property Price Predictor"
)


st.write(
    "Machine Learning based property price prediction"
)



# Sidebar model information

with st.sidebar:

    st.header(
        "🤖 Model Information"
    )


    if metrics:

        st.write(
            "Model:",
            metrics.get("model")
        )

        st.write(
            "R2 Score:",
            metrics.get("r2_score")
        )

        st.write(
            "MAE:",
            metrics.get("mae")
        )

        st.write(
            "RMSE:",
            metrics.get("rmse")
        )



# Dynamic values from dataset

locations = sorted(
    df["location"]
    .dropna()
    .unique()
)


property_types = sorted(
    df["property_type"]
    .dropna()
    .unique()
)




# Layout

input_col, graph_col = st.columns(
    [1,1]
)



# Input section

with input_col:


    st.subheader(
        "📋 Enter Property Details"
    )


    area = st.number_input(

        "Area (sq ft)",

        value=int(
            df["area"].mean()
        ),

        step=100

    )



    bedrooms = st.number_input(

        "Bedrooms",

        value=int(
            df["bedrooms"].mean()
        ),

        step=1

    )



    bathrooms = st.number_input(

        "Bathrooms",

        value=int(
            df["bathrooms"].mean()
        ),

        step=1

    )



    property_age = st.number_input(

        "Property Age",

        value=int(
            df["property_age"].mean()
        ),

        step=1

    )



    parking = st.number_input(

        "Parking",

        value=int(
            df["parking"].mean()
        ),

        step=1

    )



    location = st.selectbox(

        "Location",

        locations

    )



    property_type = st.selectbox(

        "Property Type",

        property_types

    )





# Graph section

with graph_col:


    st.subheader(
        "📊 Property Analysis"
    )


    chart_df = pd.DataFrame(

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
        figsize=(5,3)
    )


    ax.bar(

        chart_df["Feature"],

        chart_df["Value"]

    )


    ax.set_ylabel(
        "Value"
    )


    plt.xticks(
        rotation=45
    )


    plt.tight_layout()


    st.pyplot(fig)




# Prediction button

if st.button(
    "🔮 Predict Price"
):


    input_data = pd.DataFrame(

        {

        "area":[area],

        "bedrooms":[bedrooms],

        "bathrooms":[bathrooms],

        "property_age":[property_age],

        "parking":[parking],

        "location":[location],

        "property_type":[property_type]

        }

    )


    prediction = model.predict(
        input_data
    )


    price = round(
        prediction[0],
        2
    )


    st.success(

        f"💰 Estimated Price: ₹ {price} Lakhs"

    )



    # Filter properties

    similar = df[

        (df["location"]==location)

        &

        (df["property_type"]==property_type)

    ]



    tab1, tab2 = st.tabs(

        [

        "🏘 Similar Properties",

        "👤 Owner Details"

        ]

    )



    with tab1:


        st.subheader(
            "Similar Properties"
        )


        if len(similar)>0:


            display_columns = [

                col for col in similar.columns

                if "owner" not in col.lower()

                and "phone" not in col.lower()

                and "email" not in col.lower()

            ]


            st.dataframe(

                similar[display_columns]
                .head(10),

                use_container_width=True

            )


        else:

            st.info(
                "No matching properties found"
            )





    with tab2:


        st.subheader(
            "Owner Details"
        )


        if len(similar)>0:


            owner = similar.sample(
                1
            ).iloc[0]


            if "owner_name" in df.columns:

                st.write(
                    "👤 Name:",
                    owner["owner_name"]
                )


            if "owner_phone" in df.columns:

                st.write(
                    "📞 Phone:",
                    owner["owner_phone"]
                )


            if "owner_email" in df.columns:

                st.write(
                    "📧 Email:",
                    owner["owner_email"]
                )


        else:

            st.info(
                "Owner details unavailable"
            )