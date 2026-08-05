import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# Load dataset
df = pd.read_csv("housing_data.csv")


# Remove missing values
df = df.dropna()


# Features dynamically picked
X = df.drop("price", axis=1)

y = df["price"]


categorical_features = X.select_dtypes(
    include="object"
).columns.tolist()


numeric_features = X.select_dtypes(
    exclude="object"
).columns.tolist()



preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)



model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)



pipeline = Pipeline(
    [
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



X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



pipeline.fit(
    X_train,
    y_train
)



prediction = pipeline.predict(X_test)



metrics = {

"model":
"RandomForestRegressor",

"r2_score":
round(r2_score(y_test,prediction),3),

"mae":
round(mean_absolute_error(y_test,prediction),2),

"rmse":
round(
mean_squared_error(
y_test,
prediction,
squared=False
),
2
)

}



joblib.dump(
pipeline,
"model_pipeline.pkl"
)



with open(
"metrics.json",
"w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


print(
"Model trained successfully"
)

print(metrics)
