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


st.title(
    "🏠 Property Price Predictor"
)



# ---------------------------
# LOAD MODEL
# ---------------------------

try:

    model = joblib.load(
        "model.pkl"
    )


    metrics = joblib.load(
        "metrics.pkl"
    )


    st.success(
        "✅ Model loaded successfully"
    )


except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()



# ---------------------------
# SIDEBAR
# ---------------------------

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



# ---------------------------
# ENCODING
# ---------------------------


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



# ---------------------------
# PREDICT
# ---------------------------

if st.button(
    "🚀 Predict Price"
):


    if area <=0:

        st.error(
            "Area must be greater than zero"
        )


    elif bedrooms <=0:

        st.error(
            "Bedrooms must be greater than zero"
        )


    else:


        input_data = pd.DataFrame({

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

        })



        # Match training columns

        input_data = input_data[
            model.feature_names_in_
        ]



        prediction = model.predict(
            input_data
        )[0]



        st.success(
            f"💰 Estimated Price: ₹ {prediction:,.0f}"
        )



        st.info(
            f"""
📊 Expected Price Range

₹ {prediction*0.9:,.0f}

to

₹ {prediction*1.1:,.0f}
"""
        )



# ---------------------------
# GRAPH
# ---------------------------


st.subheader(
    "📈 Feature Importance"
)



importance = pd.DataFrame({

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
