import streamlit as st
import joblib
import pandas as pd


# Page config

st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="centered"
)


st.title("🏠 Property Price Predictor")


# Load model

try:

    model = joblib.load(
        "model.pkl"
    )

    st.success(
        "✅ Model Loaded Successfully"
    )


except Exception as e:

    st.error(e)
    st.stop()



# Sidebar

st.sidebar.header(
    "🤖 Model Information"
)


st.sidebar.write(
    f"""
Model:

{type(model).__name__}


Features:

{list(model.feature_names_in_)}
"""
)



# User Input

st.markdown(
    "### 📋 Enter Property Details"
)


col1, col2 = st.columns(2)


with col1:

    area = st.number_input(
        "📏 Area (sq ft)",
        min_value=0,
        value=0,
        step=100
    )


with col2:

    bedrooms = st.number_input(
        "🛏 Number of Rooms",
        min_value=0,
        value=0,
        step=1
    )



location = st.selectbox(
    "📍 Location Score",
    [1,2,3,4,5],
    index=2
)



# Prediction

if st.button(
    "🚀 Predict Price"
):


    if area <= 0:

        st.error(
            "⚠️ Area must be greater than 0"
        )


    elif bedrooms <= 0:

        st.error(
            "⚠️ Rooms must be greater than 0"
        )


    else:


        input_data = pd.DataFrame(
            [
                [
                    area,
                    bedrooms,
                    location
                ]
            ],

            columns=model.feature_names_in_
        )


        prediction = model.predict(
            input_data
        )[0]


        st.success(
            f"💰 Estimated Price: ₹ {prediction:,.0f}"
        )


        # Price range

        lower = prediction * 0.9

        upper = prediction * 1.1


        st.info(
            f"""
📊 Expected Price Range

₹ {lower:,.0f}

to

₹ {upper:,.0f}
"""
        )



        # Summary

        st.subheader(
            "🏡 Property Summary"
        )


        summary = pd.DataFrame(
            {
                "Feature":
                [
                    "Area",
                    "Rooms",
                    "Location"
                ],

                "Value":
                [
                    area,
                    bedrooms,
                    location
                ]
            }
        )


        st.table(
            summary
        )



        # Feature importance

        if hasattr(
            model,
            "feature_importances_"
        ):


            st.subheader(
                "📊 Feature Importance"
            )


            chart_data = pd.DataFrame(
                {
                    "Feature":
                    model.feature_names_in_,

                    "Importance":
                    model.feature_importances_
                }
            )


            st.bar_chart(
                chart_data.set_index(
                    "Feature"
                )
            )



# Reset button

if st.button(
    "🔄 Reset"
):

    st.rerun()


st.markdown("---")

st.caption(
    "Built using Python + Machine Learning + Streamlit 🚀"
)
