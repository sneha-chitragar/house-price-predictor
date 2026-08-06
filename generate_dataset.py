import pandas as pd
import random


locations = [
    "BTM",
    "Indiranagar",
    "Whitefield",
    "Koramangala",
    "Yelahanka",
    "Electronic City",
    "Marathahalli",
    "HSR Layout"
]


property_types = [
    "Apartment",
    "Villa",
    "Independent House"
]


owner_names = [
    "Rajesh Kumar",
    "Anil Sharma",
    "Priya Rao",
    "Sneha Patil",
    "Vikram Singh",
    "Meera Joshi"
]


data = []


for i in range(5000):

    area = random.randint(500,5000)

    bedrooms = random.randint(1,5)

    bathrooms = random.randint(1,5)

    property_age = random.randint(0,25)

    parking = random.randint(0,3)


    location = random.choice(locations)

    property_type = random.choice(property_types)



    location_rate = {

        "Indiranagar":22000,
        "Koramangala":20000,
        "HSR Layout":17000,
        "Whitefield":15000,
        "BTM":14000,
        "Marathahalli":12000,
        "Yelahanka":10000,
        "Electronic City":9000

    }


    price = (

        area * location_rate[location]

        +
        bedrooms*500000

        +
        bathrooms*200000

        -
        property_age*50000

    )


    owner=random.choice(owner_names)


    phone="9"+str(
        random.randint(
            100000000,
            999999999
        )
    )


    email=(
        owner.lower()
        .replace(" ",".")
        +"@gmail.com"
    )


    data.append([

        area,
        bedrooms,
        bathrooms,
        property_age,
        parking,
        location,
        property_type,
        round(price/100000,2),
        owner,
        phone,
        email

    ])



df=pd.DataFrame(

data,

columns=[

"area",
"bedrooms",
"bathrooms",
"property_age",
"parking",
"location",
"property_type",
"price",
"owner_name",
"owner_phone",
"owner_email"

]

)


df.to_csv(

"housing_data.csv",

index=False

)


print("Dataset created")
print(df.head())