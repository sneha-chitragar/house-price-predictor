import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


print("Training started...")


# Load dataset
df = pd.read_csv("housing_data.csv")


# Remove empty values
df = df.dropna()


# Separate features and target

X = df.drop(
    "price",
    axis=1
)

y = df["price"]



# Detect categorical columns
categorical_columns = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()



# Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)



# Model

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)



# Pipeline

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)



# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train

pipeline.fit(
    X_train,
    y_train
)



# Prediction

y_pred = pipeline.predict(
    X_test
)



# Metrics

metrics = {

    "model":
    "RandomForestRegressor",

    "r2_score":
    round(
        r2_score(
            y_test,
            y_pred
        ),
        3
    ),

    "mae":
    round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        2
    ),

  "rmse":
round(
    mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5,
    

        2
    )
}



# Save complete pipeline

joblib.dump(
    pipeline,
    "model_pipeline.pkl"
)



# Save metrics

with open(
    "metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )



print("Training completed successfully")
print(metrics)
print("Created: model_pipeline.pkl")
print("Created: metrics.json")