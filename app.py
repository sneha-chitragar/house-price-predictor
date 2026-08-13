import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bangalore Property Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# COLOR PALETTE
# ============================================================

BACKGROUND = "#0B1120"
WHITE = "#FFFFFF"
BLACK = "#111827"
BLUE = "#2563EB"
GREEN = "#16A34A"
GRAY = "#64748B"
LIGHT_GRAY = "#E2E8F0"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       MAIN BACKGROUND
       ====================================================== */

    .stApp {
        background-color: #0B1120;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 18px;
        padding-bottom: 35px;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {
        color: #FFFFFF !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
    }

    h2 {
        color: #FFFFFF !important;
        font-size: 21px !important;
        font-weight: 750 !important;
        margin-top: 12px !important;
        margin-bottom: 8px !important;
    }

    h3 {
        color: #FFFFFF !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        margin-top: 8px !important;
        margin-bottom: 6px !important;
    }


    /* ======================================================
       NORMAL TEXT
       ====================================================== */

    p {
        color: #CBD5E1 !important;
        font-size: 13px !important;
        line-height: 1.45 !important;
    }


    /* ======================================================
       CAPTIONS
       ====================================================== */

    [data-testid="stCaptionContainer"] {
        color: #94A3B8 !important;
        font-size: 11px !important;
    }

    [data-testid="stCaptionContainer"] * {
        color: #94A3B8 !important;
        font-size: 11px !important;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #263449 !important;
    }

    section[data-testid="stSidebar"] h2 {
        color: #FFFFFF !important;
        font-size: 20px !important;
    }

    section[data-testid="stSidebar"] p {
        color: #CBD5E1 !important;
        font-size: 12px !important;
    }

    section[data-testid="stSidebar"] label {
        color: #E2E8F0 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }


    /* ======================================================
       SIDEBAR INPUTS
       ====================================================== */

    section[data-testid="stSidebar"] input {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        font-size: 13px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"]
    div[data-baseweb="select"] * {
        color: #111827 !important;
        font-size: 13px !important;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    section[data-testid="stSidebar"]
    .stButton > button {

        background-color: #2563EB !important;
        color: #FFFFFF !important;

        border: none !important;
        border-radius: 9px !important;

        font-size: 13px !important;
        font-weight: 700 !important;

        min-height: 42px !important;
    }

    section[data-testid="stSidebar"]
    .stButton > button:hover {

        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    div[data-testid="stMetric"] {

        background-color: #FFFFFF !important;

        border-radius: 12px !important;

        padding: 10px 12px !important;

        border: 1px solid #E2E8F0 !important;

        box-shadow:
            0px 4px 12px
            rgba(0,0,0,0.20) !important;
    }


    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 11px !important;
    }

    div[data-testid="stMetricLabel"] * {
        color: #64748B !important;
        font-size: 11px !important;
    }


    div[data-testid="stMetricValue"] {
        color: #111827 !important;
        font-size: 20px !important;
        font-weight: 750 !important;
    }

    div[data-testid="stMetricValue"] * {
        color: #111827 !important;
        font-size: 20px !important;
    }


    /* ======================================================
       ALERTS
       ====================================================== */

    div[data-testid="stAlert"] {
        font-size: 12px !important;
        padding: 8px 12px !important;
        border-radius: 9px !important;
    }

    div[data-testid="stAlert"] p {
        font-size: 12px !important;
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {
        border-color: #263449 !important;
        margin-top: 10px !important;
        margin-bottom: 10px !important;
    }


    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD HOUSING DATA
# ============================================================

@st.cache_data
def load_housing_data():

    return pd.read_csv(
        "housing_data.csv"
    )


# ============================================================
# LOAD LOCATION DATA
# ============================================================

@st.cache_data
def load_location_data():

    if os.path.exists(
        "location_scores.csv"
    ):

        return pd.read_csv(
            "location_scores.csv"
        )

    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "model_pipeline.pkl"
    )


# ============================================================
# LOAD FILES
# ============================================================

try:

    df = load_housing_data()

except Exception as error:

    st.error(
        "Unable to load housing_data.csv"
    )

    st.exception(error)

    st.stop()


try:

    model = load_model()

except Exception as error:

    st.error(
        "Unable to load model_pipeline.pkl"
    )

    st.exception(error)

    st.stop()


location_df = load_location_data()


# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)


if location_df is not None:

    location_df.columns = (
        location_df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "area",
    "bedrooms",
    "bathrooms",
    "property_age",
    "parking",
    "price"
]


for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# ============================================================
# INDIAN NUMBER FORMAT
# ============================================================

def indian_number(value):

    try:

        value = int(
            round(
                float(value)
            )
        )

        if value < 0:

            sign = "-"

            value = abs(value)

        else:

            sign = ""


        number = str(value)


        if len(number) <= 3:

            return sign + number


        last_three = number[-3:]

        remaining = number[:-3]

        groups = []


        while len(remaining) > 2:

            groups.insert(
                0,
                remaining[-2:]
            )

            remaining = remaining[:-2]


        if remaining:

            groups.insert(
                0,
                remaining
            )


        return (
            sign
            +
            ",".join(groups)
            +
            ","
            +
            last_three
        )

    except Exception:

        return str(value)


# ============================================================
# PRICE FORMAT
# ============================================================

def format_price(price_lakhs):

    price_lakhs = float(
        price_lakhs
    )


    rupees = (
        price_lakhs
        *
        100000
    )


    if price_lakhs >= 100:

        crores = (
            price_lakhs
            /
            100
        )

        short_price = (
            f"₹{crores:.2f} Cr"
        )

    else:

        short_price = (
            f"₹{price_lakhs:.2f} L"
        )


    exact_price = (
        "₹"
        +
        indian_number(
            rupees
        )
    )


    return (
        short_price,
        exact_price,
        rupees
    )


# ============================================================
# NORMALIZE LOCATION
# ============================================================

def normalize_location(location):

    value = str(
        location
    ).strip().lower()


    value = (
        value
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
    )


    aliases = {

        "btm": "btm",

        "btmlayout": "btm",

        "whitefield": "whitefield",

        "hsr": "hsrlayout",

        "hsrlayout": "hsrlayout",

        "electroniccity":
            "electroniccity",

        "sarjapur":
            "sarjapurroad",

        "sarjapurroad":
            "sarjapurroad",

        "koramangala":
            "koramangala",

        "indiranagar":
            "indiranagar",

        "jayanagar":
            "jayanagar",

        "marathahalli":
            "marathahalli",

        "hebbal":
            "hebbal",

        "yelahanka":
            "yelahanka",

        "rajajinagar":
            "rajajinagar",

        "banashankari":
            "banashankari",

        "jpnagar":
            "jpnagar",

        "jp":
            "jpnagar",

        "bannerghattaroad":
            "bannerghattaroad",

        "kengeri":
            "kengeri"
    }


    return aliases.get(
        value,
        value
    )


# ============================================================
# LOCATION SCORE
# ============================================================

def get_location_score(
    selected_location
):

    if location_df is None:

        return None


    if "location" not in location_df.columns:

        return None


    temp = location_df.copy()


    temp["location_key"] = (
        temp["location"]
        .apply(
            normalize_location
        )
    )


    selected_key = (
        normalize_location(
            selected_location
        )
    )


    result = temp[
        temp["location_key"]
        ==
        selected_key
    ]


    if result.empty:

        return None


    row = result.iloc[0]


    def get_score(column):

        if column not in row.index:

            return 0


        try:

            return float(
                row[column]
            )

        except Exception:

            return 0


    connectivity = get_score(
        "connectivity_score"
    )

    school = get_score(
        "school_score"
    )

    metro = get_score(
        "metro_score"
    )

    hospital = get_score(
        "hospital_score"
    )

    demand = get_score(
        "demand_score"
    )


    scores = [

        connectivity,
        school,
        metro,
        hospital,
        demand

    ]


    valid_scores = [

        score
        for score in scores
        if score > 0

    ]


    if not valid_scores:

        return None


    overall = round(
        sum(valid_scores)
        /
        len(valid_scores),
        1
    )


    return {

        "overall":
            overall,

        "connectivity":
            connectivity,

        "school":
            school,

        "metro":
            metro,

        "hospital":
            hospital,

        "demand":
            demand
    }


# ============================================================
# HEADER
# ============================================================

st.title(
    "🏠 Bangalore Property Intelligence"
)

st.caption(
    "AI-Powered Property Valuation & Location Intelligence System"
)


# ============================================================
# DASHBOARD FEATURES
# ============================================================

st.subheader(
    "✨ Dashboard Features"
)


feature1, feature2, feature3, feature4 = (
    st.columns(4)
)


with feature1:

    st.metric(
        "💰 Valuation",
        "AI Powered"
    )

    st.caption(
        "Predict property market value."
    )


with feature2:

    st.metric(
        "📍 Location",
        "Intelligence"
    )

    st.caption(
        "Analyze locality quality."
    )


with feature3:

    st.metric(
        "📊 Market",
        "Analysis"
    )

    st.caption(
        "Compare Bangalore localities."
    )


with feature4:

    st.metric(
        "📈 Insights",
        "Dynamic"
    )

    st.caption(
        "Charts change with inputs."
    )


st.info(
    "👈 Enter your property details in the left panel "
    "and click **Estimate Property Price**."
)


st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🔎 Property Details"
    )

    st.caption(
        "Enter property details to estimate "
        "the current market value."
    )


    # ========================================================
    # AREA
    # ========================================================

    area_values = (
        df["area"]
        .dropna()
    )


    if not area_values.empty:

        area_min = max(
            100,
            int(
                area_values.min()
            )
        )

        area_max = max(
            area_min + 100,
            int(
                area_values.max()
            )
        )

        area_default = int(
            area_values.median()
        )

    else:

        area_min = 100

        area_max = 10000

        area_default = 1200


    area = st.number_input(
        "Area (sq.ft.)",
        min_value=area_min,
        max_value=area_max,
        value=area_default,
        step=50
    )


    # ========================================================
    # BEDROOMS
    # ========================================================

    bedroom_values = sorted(

        df["bedrooms"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()

    )


    bedrooms = st.selectbox(
        "Bedrooms",
        bedroom_values
    )


    # ========================================================
    # BATHROOMS
    # ========================================================

    bathroom_values = sorted(

        df["bathrooms"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()

    )


    bathrooms = st.selectbox(
        "Bathrooms",
        bathroom_values
    )


    # ========================================================
    # PROPERTY AGE
    # ========================================================

    property_age = st.number_input(
        "Property Age (years)",
        min_value=0,
        max_value=100,
        value=5,
        step=1
    )


    # ========================================================
    # PARKING
    # ========================================================

    parking = st.selectbox(
        "Parking",
        [0, 1, 2, 3]
    )


    # ========================================================
    # LOCATION
    # ========================================================

    location_values = sorted(

        df["location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()

    )


    location = st.selectbox(
        "Location",
        location_values
    )


    # ========================================================
    # PROPERTY TYPE
    # ========================================================

    property_type_values = sorted(

        df["property_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()

    )


    property_type = st.selectbox(
        "Property Type",
        property_type_values
    )


    st.divider()


    estimate_button = st.button(
        "🔮 Estimate Property Price",
        use_container_width=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:

    st.session_state.prediction = None


# ============================================================
# RUN PREDICTION
# ============================================================

if estimate_button:

    input_data = pd.DataFrame(
        {
            "area": [
                area
            ],

            "bedrooms": [
                bedrooms
            ],

            "bathrooms": [
                bathrooms
            ],

            "property_age": [
                property_age
            ],

            "parking": [
                parking
            ],

            "location": [
                location
            ],

            "property_type": [
                property_type
            ]
        }
    )


    try:

        prediction = model.predict(
            input_data
        )[0]


        st.session_state.prediction = float(
            prediction
        )


    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(
            error
        )

        st.stop()


# ============================================================
# INITIAL SCREEN
# ============================================================

if st.session_state.prediction is None:

    st.header(
        "🏠 Property Valuation Dashboard"
    )

    st.write(
        "Enter property details using the controls "
        "available in the left panel."
    )

    st.write(
        "The application will calculate the estimated "
        "market value and show location and market insights."
    )

    st.info(
        "Start by entering Area, Bedrooms, Bathrooms, "
        "Property Age, Parking, Location and Property Type."
    )

    st.stop()


# ============================================================
# GET PREDICTION
# ============================================================

prediction = (
    st.session_state.prediction
)


short_price, exact_price, rupees = (
    format_price(
        prediction
    )
)


price_per_sqft = (
    rupees
    /
    area
)


# ============================================================
# ESTIMATED PROPERTY VALUE
# ============================================================

st.header(
    "💰 Estimated Property Value"
)


price_column, sqft_column = (
    st.columns([2, 1])
)


with price_column:

    st.metric(
        "Estimated Market Price",
        short_price
    )

    st.caption(
        f"Exact estimated value: {exact_price}"
    )


with sqft_column:

    st.metric(
        "Price per sq.ft.",
        f"₹{indian_number(price_per_sqft)}"
    )


st.divider()


# ============================================================
# PROPERTY SUMMARY
# ============================================================

st.header(
    "🔎 Property Summary"
)


summary1, summary2, summary3, summary4, summary5 = (
    st.columns(5)
)


with summary1:

    st.metric(
        "📐 Area",
        f"{area:,} sq.ft."
    )


with summary2:

    st.metric(
        "🛏 Bedrooms",
        bedrooms
    )


with summary3:

    st.metric(
        "🚿 Bathrooms",
        bathrooms
    )


with summary4:

    st.metric(
        "📍 Location",
        location
    )


with summary5:

    st.metric(
        "🏢 Type",
        property_type
    )


st.divider()


# ============================================================
# LOCATION INTELLIGENCE
# ============================================================

st.header(
    "📍 Location Intelligence"
)


location_score = get_location_score(
    location
)


if location_score is None:

    st.warning(
        f"No location score available for {location}."
    )

else:

    overall = location_score[
        "overall"
    ]


    if overall >= 9:

        status = (
            "Outstanding Location"
        )

    elif overall >= 8:

        status = (
            "Excellent Location"
        )

    elif overall >= 7:

        status = (
            "Very Good Location"
        )

    elif overall >= 6:

        status = (
            "Good Location"
        )

    else:

        status = (
            "Average Location"
        )


    score_column, factor_column = (
        st.columns([1, 4])
    )


    with score_column:

        st.metric(
            "⭐ Location Score",
            f"{overall:.1f} / 10"
        )

        st.success(
            status
        )


    with factor_column:

        factor1, factor2, factor3, factor4, factor5 = (
            st.columns(5)
        )


        with factor1:

            st.metric(
                "🛣 Connectivity",
                f"{location_score['connectivity']:.1f}"
            )


        with factor2:

            st.metric(
                "🏫 Schools",
                f"{location_score['school']:.1f}"
            )


        with factor3:

            st.metric(
                "🚇 Metro",
                f"{location_score['metro']:.1f}"
            )


        with factor4:

            st.metric(
                "🏥 Hospitals",
                f"{location_score['hospital']:.1f}"
            )


        with factor5:

            st.metric(
                "📈 Demand",
                f"{location_score['demand']:.1f}"
            )


st.divider()


# ============================================================
# MARKET ANALYSIS
# ============================================================

st.header(
    "📊 Market Value Analysis"
)

st.caption(
    "The selected property is highlighted in the graphs."
)


graph_column1, graph_column2 = (
    st.columns(2)
)


# ============================================================
# SCATTER GRAPH
# ============================================================

with graph_column1:

    st.markdown(
        "#### 🔵 Area vs Property Price"
    )


    if (
        "area" in df.columns
        and
        "price" in df.columns
    ):

        graph_data = df[
            [
                "area",
                "price"
            ]
        ].copy()


        graph_data["area"] = pd.to_numeric(
            graph_data["area"],
            errors="coerce"
        )


        graph_data["price"] = pd.to_numeric(
            graph_data["price"],
            errors="coerce"
        )


        graph_data = (
            graph_data
            .dropna()
        )


        if not graph_data.empty:

            fig, ax = plt.subplots(
                figsize=(5.5, 3.4)
            )


            # Existing properties

            ax.scatter(

                graph_data["area"],

                graph_data["price"],

                alpha=0.35,

                s=25

            )


            # Selected property

            ax.scatter(

                area,

                prediction,

                s=220,

                marker="*",

                color=GREEN,

                edgecolors=BLACK,

                linewidths=1.2,

                zorder=10

            )


            ax.set_title(

                "Area vs Property Price",

                fontsize=12,

                fontweight="bold"

            )


            ax.set_xlabel(

                "Area (sq.ft.)",

                fontsize=9

            )


            ax.set_ylabel(

                "Price (Lakhs)",

                fontsize=9

            )


            ax.tick_params(
                labelsize=8
            )


            ax.grid(
                alpha=0.2
            )


            fig.tight_layout()


            st.pyplot(
                fig,
                use_container_width=True
            )


            plt.close(fig)


            st.caption(
                "⭐ Green star = selected property"
            )


# ============================================================
# BAR GRAPH
# ============================================================

with graph_column2:

    st.markdown(
        "#### 🟢 Average Price by Locality"
    )


    if (
        "location" in df.columns
        and
        "price" in df.columns
    ):

        locality_prices = (

            df.groupby(
                "location"
            )["price"]

            .mean()

            .sort_values(
                ascending=False
            )

            .head(8)

        )


        selected_rows = df[
            df["location"]
            .astype(str)
            .str.strip()
            .str.lower()
            ==
            str(location)
            .strip()
            .lower()
        ]


        if not selected_rows.empty:

            selected_average = (
                selected_rows[
                    "price"
                ].mean()
            )


            selected_exists = any(

                str(item)
                .strip()
                .lower()
                ==
                str(location)
                .strip()
                .lower()

                for item
                in locality_prices.index

            )


            if not selected_exists:

                locality_prices.loc[
                    location
                ] = selected_average


        if not locality_prices.empty:

            fig, ax = plt.subplots(
                figsize=(5.5, 3.4)
            )


            bar_colors = []


            for item in locality_prices.index:

                if (

                    str(item)
                    .strip()
                    .lower()
                    ==
                    str(location)
                    .strip()
                    .lower()

                ):

                    bar_colors.append(
                        GREEN
                    )

                else:

                    bar_colors.append(
                        BLUE
                    )


            bars = ax.bar(

                locality_prices.index,

                locality_prices.values,

                color=bar_colors

            )


            ax.set_title(

                "Average Price by Locality",

                fontsize=12,

                fontweight="bold"

            )


            ax.set_xlabel(

                "Locality",

                fontsize=9

            )


            ax.set_ylabel(

                "Average Price (Lakhs)",

                fontsize=9

            )


            ax.tick_params(

                axis="x",

                rotation=35,

                labelsize=7

            )


            ax.tick_params(

                axis="y",

                labelsize=8

            )


            ax.grid(

                axis="y",

                alpha=0.2

            )


            # Values above bars

            for bar in bars:

                height = (
                    bar.get_height()
                )


                ax.text(

                    bar.get_x()
                    +
                    bar.get_width()
                    /
                    2,

                    height,

                    f"{height:.1f}",

                    ha="center",

                    va="bottom",

                    fontsize=7

                )


            fig.tight_layout()


            st.pyplot(

                fig,

                use_container_width=True

            )


            plt.close(fig)


            st.caption(
                "🟢 Green = selected locality"
            )


st.divider()


# ============================================================
# MARKET POSITION
# ============================================================

st.header(
    "💡 Locality Market Position"
)


selected_locality_data = df[

    df["location"]
    .astype(str)
    .str.strip()
    .str.lower()

    ==

    str(location)
    .strip()
    .lower()

]


if not selected_locality_data.empty:

    locality_average = (
        selected_locality_data[
            "price"
        ].mean()
    )


    locality_price, _, _ = (
        format_price(
            locality_average
        )
    )


    difference = (
        prediction
        -
        locality_average
    )


    if locality_average != 0:

        difference_percent = (

            difference
            /
            locality_average

        ) * 100

    else:

        difference_percent = 0


    market1, market2, market3 = (
        st.columns(3)
    )


    with market1:

        st.metric(
            "📍 Locality Average",
            locality_price
        )


    with market2:

        st.metric(
            "🏠 Your Property",
            short_price
        )


    with market3:

        st.metric(
            "📊 Difference",
            f"{difference_percent:+.1f}%"
        )


else:

    st.info(
        "Locality market comparison is not available."
    )


# ============================================================
# OWNER INFORMATION
# ============================================================

owner_columns = [

    "owner_name",

    "owner_phone",

    "owner_email"

]


if all(

    column in df.columns

    for column
    in owner_columns

):

    st.divider()


    st.header(
        "👤 Owner Information"
    )


    owner_data = df[

        df["location"]
        .astype(str)
        .str.strip()
        .str.lower()

        ==

        str(location)
        .strip()
        .lower()

    ]


    if not owner_data.empty:

        owner = owner_data.iloc[0]


        owner1, owner2, owner3 = (
            st.columns(3)
        )


        with owner1:

            st.metric(
                "👤 Owner",
                str(
                    owner[
                        "owner_name"
                    ]
                )
            )


        with owner2:

            st.metric(
                "📞 Phone",
                str(
                    owner[
                        "owner_phone"
                    ]
                )
            )


        with owner3:

            st.metric(
                "✉ Email",
                str(
                    owner[
                        "owner_email"
                    ]
                )
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "🏠 Bangalore Property Intelligence | "
    "AI Property Valuation | "
    "Location Intelligence | "
    "Bangalore Market Analysis"
)