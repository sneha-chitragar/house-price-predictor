import pandas as pd
import numpy as np

np.random.seed(42)

# Real Bangalore pricing (₹ per sq.ft converted to Lakhs approx)
location_price_factor = {
    "Indiranagar": 0.12,
    "Whitefield": 0.09,
    "HSR Layout": 0.11,
    "BTM": 0.08,
    "Electronic City": 0.07,
    "Marathahalli": 0.085,
    "Hebbal": 0.10,
    "Yelahanka": 0.065,
    "Kengeri": 0.06,
    "Bannerghatta": 0.075
}

rows = []

for _ in range(1200):
    area = np.random.randint(500, 5000)
    bedrooms = np.random.randint(1, 6)
    location = np.random.choice(list(location_price_factor.keys()))

    base_price = area * location_price_factor[location]

    price = (
        base_price +
        bedrooms * 8 +
        np.random.randint(-10, 10)
    )

    rows.append([area, bedrooms, location, round(price, 2)])

df = pd.DataFrame(rows, columns=[
    "area",
    "bedrooms",
    "location",
    "price"
])

df.to_csv("housing.csv", index=False)

print("✅ Realistic Bangalore dataset created")
