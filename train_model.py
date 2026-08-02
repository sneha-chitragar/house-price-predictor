import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

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

# Train model
model = RandomForestRegressor(random_state=42)

model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("New property price model created")
