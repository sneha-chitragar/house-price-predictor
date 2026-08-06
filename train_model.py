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



print("Training started")


df=pd.read_csv(
"housing_data.csv"
)


print(df.columns)



X=df.drop(

columns=[

"price",
"owner_name",
"owner_phone",
"owner_email"

]

)


y=df["price"]



categorical=X.select_dtypes(
include="object"
).columns.tolist()



preprocessor=ColumnTransformer(

[

(
"cat",
OneHotEncoder(
handle_unknown="ignore"
),
categorical
)

],

remainder="passthrough"

)



model=RandomForestRegressor(

n_estimators=300,

random_state=42

)



pipeline=Pipeline(

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



X_train,X_test,y_train,y_test=train_test_split(

X,
y,
test_size=0.2,
random_state=42

)



pipeline.fit(

X_train,
y_train

)



prediction=pipeline.predict(

X_test

)



rmse=mean_squared_error(

y_test,
prediction

)**0.5



metrics={

"model":"RandomForestRegressor",

"r2_score":round(
r2_score(
y_test,
prediction
),
3
),

"mae":round(
mean_absolute_error(
y_test,
prediction
),
2
),

"rmse":round(
rmse,
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



print(metrics)

print("Model created")