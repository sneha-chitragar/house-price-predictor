import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv("Bangalore_House_Data.csv")

print("✅ Dataset Loaded")
print(df.head())

# ==============================
# 2. BASIC CLEANING
# ==============================

# Drop rows with missing values
df = df.dropna()

# Convert total_sqft (if string ranges like "1000-1200")
def convert_sqft(x):
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None

df['area'] = df['area'].apply(convert_sqft)

# Drop again if conversion failed
df = df.dropna()

# ==============================
# 3. FEATURE SELECTION
# ==============================

# Keep only required columns
df = df[['area', 'bedrooms', 'bathrooms', 'parking', 'location', 'price']]

# ==============================
# 4. ONE-HOT ENCODING (LOCATION)
# ==============================

df = pd.get_dummies(df, columns=['location'])

print("✅ After Encoding:", df.shape)

# ==============================
# 5. SPLIT DATA
# ==============================

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. TRAIN MODEL
# ==============================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model Training Completed")

# ==============================
# 7. EVALUATION
# ==============================

y_pred = model.predict(X_test)

print("📊 R2 Score:", r2_score(y_test, y_pred))
print("📊 MAE:", mean_absolute_error(y_test, y_pred))

# ==============================
# 8. SAVE MODEL + COLUMNS
# ==============================

joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ New property price model created")
print("✅ model.pkl and columns.pkl saved successfully")
