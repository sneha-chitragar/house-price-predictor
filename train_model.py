import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Sample dataset
data = {
    "area": [800, 1000, 1200, 1500, 1800, 2200],
    "bedrooms": [1, 2, 2, 3, 3, 4],
    "location": [2, 3, 4, 4, 5, 5],
    "price": [4000000, 5500000, 7000000, 9000000, 12000000, 15000000]
}


df = pd.DataFrame(data)


# Features and Target

X = df[
    [
        "area",
        "bedrooms",
        "location"
    ]
]


y = df["price"]


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# Evaluation

prediction = model.predict(X_test)


metrics = {

    "r2": r2_score(y_test, prediction),

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


# Save model

joblib.dump(
    model,
    "model.pkl"
)


# Save metrics

joblib.dump(
    metrics,
    "metrics.pkl"
)


print("Model trained successfully")
print(metrics)
