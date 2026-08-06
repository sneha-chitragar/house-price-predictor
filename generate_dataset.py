import pandas as pd
import random


locations = [
    "BTM",
    "Indiranagar",
    "Yelahanka",
    "Whitefield",
    "Koramangala",
    "Electronic City"
]


property_types = [
    "Apartment",
    "Villa",
    "Independent House"
]


data = []


for i in range(1000):

    area = random.randint(500, 5000)

    bedrooms = random.randint(1, 5)

    bathrooms = random.randint(1, 5)

    location = random.choice(
        locations
    )

    property_age = random.randint(
        0,
        30
    )

    parking = random.randint(
        0,
        2
    )

    property_type = random.choice(
        property_types
    )


    # Location price factor

    location_factor = {

        "BTM": 8000,

        "Indiranagar": 15000,

        "Yelahanka": 9000,

        "Whitefield": 12000,

        "Koramangala": 18000,

        "Electronic City": 7000

    }


    price = (
            area * location_factor[location]
    + bedrooms * 800000
    + bathrooms * 500000
    + parking * 300000
    - property_age * 100000
    )


    # Property type impact

    if property_type == "Villa":
        price += 3000000

    elif property_type == "Independent House":
        price += 2000000


    data.append(
        [
            area,
            bedrooms,
            location,
            bathrooms,
            property_age,
            parking,
            property_type,
            price
        ]
    )



df = pd.DataFrame(
    data,
    columns=[
        "area",
        "bedrooms",
        "location",
        "bathrooms",
        "property_age",
        "parking",
        "property_type",
        "price"
    ]
)


df.to_csv(
    "housing_data.csv",
    index=False
)


print(
    "New property dataset created successfully"
)