import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample dataset
data = pd.DataFrame({
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 5],
    'location_score': [1, 2, 3, 4, 5],
    'price': [50, 80, 120, 160, 200]
})

X = data[['area', 'bedrooms', 'location_score']]
y = data['price']

model = RandomForestRegressor()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')

print("Model saved successfully ✅")
