import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Running UPDATED train_model.py")
# Create sample property data
data = {
    "area": [800,1000,1200,1500,1800,2200,2500],
    "bedrooms": [1,2,2,3,3,4,5],
    "location": [1,2,3,3,4,4,5],
    "price": [2000000,3500000,4500000,6000000,7500000,10000000,15000000]
}

df = pd.DataFrame(data)


# Features
X = df[["area", "bedrooms", "location"]]

# Target
y = df["price"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train model
model = RandomForestRegressor(
    random_state=42
)

model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Model evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(
    y_test,
    y_pred,
    squared=False
)


print("R² Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# Save model
joblib.dump(model, "model.pkl")


# Save metrics
metrics = {
    "r2": r2,
    "mae": mae,
    "rmse": rmse
}

joblib.dump(metrics, "metrics.pkl")


# Save feature importance
feature_importance = {
    "Area": model.feature_importances_[0],
    "Bedrooms": model.feature_importances_[1],
    "Location": model.feature_importances_[2]
}

joblib.dump(
    feature_importance,
    "feature_importance.pkl"
)


print("✅ Model, metrics and feature importance saved successfully")

