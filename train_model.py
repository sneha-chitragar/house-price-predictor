import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv("housing.csv")

# One-hot encoding for location
data = pd.get_dummies(data, columns=["location"])

# Features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split (for better evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("📊 Model Performance:")
print(f"R2 Score: {round(r2, 3)}")
print(f"MAE: {round(mae, 2)} Lakhs")

# Save model
joblib.dump(model, "model.pkl")

# Save column structure (VERY IMPORTANT for app.py)
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Model trained and saved successfully")
