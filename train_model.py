import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# -------------------------
# DATA
# -------------------------

data = {

"area":[800,1000,1200,1500,1800,2200,2500,3000],

"bedrooms":[1,2,2,3,3,4,4,5],

"bathrooms":[1,2,2,3,3,4,4,5],

"parking":[0,1,1,1,2,2,2,3],

"age":[20,15,10,8,5,3,2,1],

"property_type":[
"Apartment",
"Apartment",
"Villa",
"Apartment",
"Villa",
"Independent House",
"Villa",
"Independent House"
],

"location":[
"Electronic City",
"Whitefield",
"HSR Layout",
"Marathahalli",
"Koramangala",
"Whitefield",
"HSR Layout",
"Koramangala"
],

"price":[
4000000,
5500000,
7500000,
9500000,
12000000,
15000000,
18000000,
25000000
]

}


df=pd.DataFrame(data)



# -------------------------
# ENCODING
# -------------------------

property_encoder=LabelEncoder()

location_encoder=LabelEncoder()



df["property_type"]=property_encoder.fit_transform(
    df["property_type"]
)


df["location"]=location_encoder.fit_transform(
    df["location"]
)



# SAVE ENCODERS

joblib.dump(
    property_encoder,
    "property_encoder.pkl"
)


joblib.dump(
    location_encoder,
    "location_encoder.pkl"
)



# -------------------------
# MODEL
# -------------------------

X=df.drop(
    "price",
    axis=1
)


y=df["price"]



X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model=RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



# -------------------------
# METRICS
# -------------------------

prediction=model.predict(
    X_test
)


metrics={

"r2":r2_score(
    y_test,
    prediction
),

"mae":mean_absolute_error(
    y_test,
    prediction
),

"rmse":mean_squared_error(
    y_test,
    prediction,
    squared=False
)

}



# SAVE MODEL

joblib.dump(
    model,
    "model.pkl"
)


joblib.dump(
    metrics,
    "metrics.pkl"
)



print("Training completed")
