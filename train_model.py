import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Bangalore_House_Data.csv")

# Keep required columns
df = df[['area', 'bedrooms', 'bathrooms', 'parking', 'location', 'price']]

# Convert location → numbers
df = pd.get_dummies(df, columns=['location'])

# Features & target
X = df.drop("price", axis=1)
y = df["price"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ New property price model created")
