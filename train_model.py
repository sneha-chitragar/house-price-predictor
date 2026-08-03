import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ---------------------------
# SAMPLE PROPERTY DATA
# ---------------------------

data = {

    "area": [
        800,1000,1200,1500,1800,
        2200,2500,3000,3500,4000
    ],

    "bedrooms": [
        1,2,2,3,3,
        4,4,5,5,6
    ],

    "bathrooms": [
        1,2,2,3,3,
        4,4,5,5,6
    ],

    "parking": [
        0,1,1,1,2,
        2,2,3,3,4
    ],

    "age": [
        20,15,10,8,5,
        4,3,2,1,0
    ],

    "property_type": [
        0,0,1,0,1,
        2,1,2,1,2
    ],

    "location": [
        0,1,2,3,4,
        1,2,4,3,0
    ],


    "price": [
        4000000,
        5500000,
        7000000,
        9000000,
        12000000,
        15000000,
        18000000,
        22000000,
        28000000,
        35000000
    ]
}


df = pd.DataFrame(data)



# ---------------------------
# FEATURES
# ---------------------------

X = df.drop(
    "price",
    axis=1
)


y = df["price"]



# ---------------------------
# TRAIN TEST SPLIT
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# ---------------------------
# MODEL
# ---------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)



# ---------------------------
# EVALUATION
# ---------------------------

prediction = model.predict(
    X_test
)


metrics = {

    "r2": r2_score(
        y_test,
        prediction
    ),

    "mae": mean_absolute_error(
        y_test,
        prediction
    ),

    "rmse": mean_squared_error(
        y_test,
        prediction,
        squared=False
    )
}



# ---------------------------
# SAVE FILES
# ---------------------------

joblib.dump(
    model,
    "model.pkl"
)


joblib.dump(
    metrics,
    "metrics.pkl"
)



print("Model training completed")
print(metrics)
