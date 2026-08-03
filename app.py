import streamlit as st
import joblib
import pandas as pd


# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)


st.title("🏠 Property Price Predictor")


# -------------------------
# LOAD MODEL
# -------------------------

try:

    model = joblib.load(
        "model.pkl"
    )

    st.success(
        "✅ Model loaded successfully"
    )


except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()



# -------------------------
# LOAD METRICS
# -------------------------

try:

    metrics = joblib.load(
        "metrics.pkl"
    )


    st.sidebar.header(
        "📊 Model Performance"
    )


    st.sidebar.metric(
        "R2 Score",
        round(metrics["r2"],2)
    )


    st.sidebar.metric(
        "MAE",
        f"₹ {metrics['mae']:,.0f}"
    )


    st.sidebar.metric(
        "RMSE",
        f"₹ {metrics['rmse']:,.0f}"
    )


except:

    pass



# -------------------------
# INPUT SECTION
# -------------------------

st.header(
    "🏠 Enter Property Details"
)


col1,col2 = st.columns(2)


with col1:

    area = st.number_input(
        "📏 Area (sq ft)",
        min_value=0,
        step=100
    )


    bedrooms = st.number_input(
        "🛏 Bedrooms",
        min_value=0,
        step=1
    )


    bathrooms = st.number_input(
        "🚿 Bathrooms",
        min_value=0,
        step=1
    )


with col2:

    age = st.number_input(
        "🏚 Property Age",
        min_value=0,
        step=1
    )


    parking = st.selectbox(
        "🚗 Parking",
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



# -------------------------
# ENCODING
# -------------------------

property_mapping = {

    "Apartment":0,
    "Villa":1,
    "Independent House":2

}



location_mapping = {

    "Electronic City":0,
    "Whitefield":1,
    "HSR Layout":2,
    "Marathahalli":3,
    "Koramangala":4

}



# -------------------------
# PREDICTION
# -------------------------

if st.button(
    "🚀 Predict Price"
):


    if area <=0:

        st.error(
            "⚠️ Area must be greater than 0"
        )


    elif bedrooms <=0:

        st.error(
            "⚠️ Bedrooms must be greater than 0"
        )


    else:


        input_data = pd.DataFrame(

            [[

                area,

                bedrooms,

                bathrooms,

                1 if parking=="Yes" else 0,

                age,

                property_mapping[property_type],

                location_mapping[location]

            ]],

            columns=model.feature_names_in_

        )



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

₹ {prediction*0.9:,.0f}

to

₹ {prediction*1.1:,.0f}
"""
        )



# -------------------------
# GRAPH
# -------------------------

if hasattr(
    model,
    "feature_importances_"
):


    st.subheader(
        "📈 Feature Importance"
    )


    graph_data = pd.DataFrame({

        "Feature":
        model.feature_names_in_,


        "Importance":
        model.feature_importances_

    })


    st.bar_chart(
        graph_data.set_index(
            "Feature"
        )
    )
