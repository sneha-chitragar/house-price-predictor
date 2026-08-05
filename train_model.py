import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print("🚀 Script started")

# Show current folder
print("📁 Current working directory:", os.getcwd())

# ==============================
# LOAD DATA
# ==============================
try:
    df = pd.read_csv("Bangalore_House_Data.csv")
    print("✅ Dataset loaded")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

print(df.head())

# ==============================
# CHECK COLUMNS
# ==============================
required_cols = ['area', 'bedrooms', 'bathrooms', 'parking', 'location', 'price']

for col in required_cols:
    if col not in df.columns:
        print(f"❌ Missing column: {col}")
        exit()

# ==============================
# PREPROCESSING
# ==============================
df = df[required_cols]

df = pd.get_dummies(df, columns=['location'])

# ==============================
# FEATURES & TARGET
# ==============================
X = df.drop("price", axis=1)
y = df["price"]

# ==============================
# TRAIN MODEL
# ==============================
model = RandomForestRegressor(n_estimators=100)

model.fit(X, y)

print("✅ Model trained successfully")

# ==============================
# SAVE FILES
# ==============================
try:
    joblib.dump(model, "model.pkl")
    print("✅ model.pkl saved")
except Exception as e:
    print("❌ Error saving model:", e)

try:
    joblib.dump(X.columns.tolist(), "columns.pkl")
    print("✅ columns.pkl saved")
except Exception as e:
    print("❌ Error saving columns:", e)

print("🎉 DONE")
