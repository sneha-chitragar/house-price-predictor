from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import joblib

# After prediction
y_pred = pipeline.predict(X_test)

metrics = {
    "model": "RandomForestRegressor",
    "r2_score": round(r2_score(y_test, y_pred), 3),
    "mae": round(mean_absolute_error(y_test, y_pred), 2),
    "rmse": round(
        mean_squared_error(y_test, y_pred, squared=False),
        2
    )
}


# Save model
joblib.dump(
    pipeline,
    "model_pipeline.pkl"
)


# Save metrics
with open("metrics.json", "w") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


print("Model and metrics saved successfully")
print(metrics)
