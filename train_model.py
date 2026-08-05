import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("Bangalore_House_Data.csv")

# Keep only required columns
df = df[['area', 'bedrooms', 'bathrooms', 'parking', 'location', 'price']]

# Convert location to numeric (one-hot encoding)
df = pd.get_dummies(df, columns=['location'])

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

print("✅ Model trained")

# ==============================
# SAVE FILES (IMPORTANT)
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ model.pkl saved")
print("✅ columns.pkl saved")
