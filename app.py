import streamlit as st
import pandas as pd
import joblib
import json



st.set_page_config(
page_title="Property Price Predictor",
layout="wide"
)



model = joblib.load(
"model_pipeline.pkl"
)



df = pd.read_csv(
"housing_data.csv"
)



with open(
"metrics.json"
) as f:
    metrics=json.load(f)



st.title(
"🏠 Property Price Predictor"
)



st.sidebar.header(
"Model Information"
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



input_data={}



for column in df.drop(
"price",
axis=1
).columns:


    if df[column].dtype=="object":

        input_data[column]=st.selectbox(
            column,
            sorted(
                df[column]
                .dropna()
                .unique()
            )
        )


    else:

        input_data[column]=st.number_input(
            column,
            float(
                df[column].min()
            ),
            float(
                df[column].max()
            )
        )



input_df=pd.DataFrame(
[input_data]
)



if st.button(
"Predict Price"
):

    result=model.predict(
        input_df
    )


    st.success(
        f"Estimated Price: ₹ {result[0]:,.2f}"
    )
