import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Bangalore Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)


# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load("model_pipeline.pkl")


@st.cache_data
def load_housing_data():

    df = pd.read_csv("housing_data.csv")

    df.columns = df.columns.str.strip()

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
    )

    return df



@st.cache_data
def load_location_scores():

    score_df = pd.read_csv(
        "location_scores.csv"
    )

    score_df.columns = (
        score_df.columns
        .str.strip()
    )

    score_df["location_key"] = (
        score_df["location"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return score_df



model = load_model()

df = load_housing_data()

score_df = load_location_scores()



# ---------------- METRICS ----------------

try:

    with open("metrics.json","r") as f:
        metrics=json.load(f)

except:

    metrics={}



# ---------------- HEADER ----------------


st.markdown(
"""
<h1 style='text-align:center'>
🏠 Bangalore Property Price Predictor
</h1>

<p style='text-align:center'>
AI based Property Price Prediction + Location Analytics
</p>
""",
unsafe_allow_html=True
)



# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.header("🤖 Model Information")

    st.metric(
        "Model",
        metrics.get(
            "model",
            "RandomForest"
        )
    )

    st.metric(
        "R2 Score",
        metrics.get(
            "r2_score",
            "N/A"
        )
    )



# ---------------- INPUT ----------------


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



col1,col2 = st.columns(2)



with col1:

    st.subheader(
        "📋 Property Details"
    )


    area = st.number_input(
        "Area Sq Ft",
        300,
        5000,
        1200
    )


    bedrooms = st.slider(
        "Bedrooms",
        1,
        10,
        2
    )


    bathrooms = st.slider(
        "Bathrooms",
        1,
        10,
        2
    )


    property_age = st.slider(
        "Property Age",
        0,
        50,
        5
    )


    parking = st.slider(
        "Parking",
        0,
        5,
        1
    )


    location = st.selectbox(
        "📍 Location",
        locations
    )


    property_type = st.selectbox(
        "🏢 Property Type",
        property_types
    )


    predict_btn = st.button(
        "🔮 Predict Price"
    )




with col2:

    st.subheader(
        "📊 Property Visualization"
    )


    chart=pd.DataFrame(
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


    fig=px.bar(
        chart,
        x="Feature",
        y="Value"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



# ---------------- PREDICTION ----------------


if predict_btn:


    input_data=pd.DataFrame(
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


    try:

        price=model.predict(
            input_data
        )[0]


        st.success(
            f"💰 Estimated Price ₹ {round(price,2)} Lakhs"
        )


    except Exception as e:

        st.error(e)




    # ---------------- LOCATION SCORE ----------------


    st.subheader(
        "📍 Location Intelligence"
    )


    selected_key = (
        location
        .strip()
        .lower()
    )


    location_result = score_df[
        score_df["location_key"]
        ==
        selected_key
    ]



    if len(location_result)>0:


        row = location_result.iloc[0]


        connectivity = row["connectivity_score"]

        schools = row["school_score"]

        metro = row["metro_score"]

        hospitals = row["hospital_score"]



        a,b,c,d = st.columns(4)


        with a:

            st.metric(
                "🚗 Connectivity",
                f"{connectivity}/10"
            )


        with b:

            st.metric(
                "🏫 Schools",
                f"{schools}/10"
            )


        with c:

            st.metric(
                "🚇 Metro",
                f"{metro}/10"
            )


        with d:

            st.metric(
                "🏥 Hospitals",
                f"{hospitals}/10"
            )



        graph=pd.DataFrame(
            {
            "Category":
            [
                "Connectivity",
                "Schools",
                "Metro",
                "Hospitals"
            ],

            "Score":
            [
                connectivity,
                schools,
                metro,
                hospitals
            ]
            }
        )


        fig_score=px.bar(
            graph,
            x="Category",
            y="Score",
            text="Score",
            title=f"{location} Location Score"
        )


        st.plotly_chart(
            fig_score,
            use_container_width=True
        )


    else:


        st.warning(
            f"No location score available for {location}"
        )



    # ---------------- SIMILAR PROPERTY ----------------


    tab1,tab2=st.tabs(
        [
            "🏘 Similar Properties",
            "👤 Owner Details"
        ]
    )


    similar=df[
        (df["location"]==location)
        &
        (df["property_type"]==property_type)
    ]



    with tab1:


        if len(similar)>0:

            st.dataframe(
                similar.head(10),
                use_container_width=True
            )

        else:

            st.info(
                "No similar properties found"
            )



    with tab2:


        if len(similar)>0:


            owner=similar.iloc[0]


            st.write(
                "👤",
                owner.get(
                    "owner_name",
                    "Not Available"
                )
            )


            st.write(
                "📞",
                owner.get(
                    "owner_phone",
                    "Not Available"
                )
            )


            st.write(
                "📧",
                owner.get(
                    "owner_email",
                    "Not Available"
                )
            )


        else:

            st.info(
                "Owner details unavailable"
            )