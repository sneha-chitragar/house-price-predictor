import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "housing_data.csv"
MODEL_FILE = BASE_DIR / "model_pipeline.pkl"
METRICS_FILE = BASE_DIR / "metrics.json"


# ============================================================
# FIND PRICE COLUMN
# ============================================================

PRICE_COLUMNS = [
    "price",
    "Price",
    "PRICE"
]


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("\n==============================")
    print("HOUSE PRICE MODEL TRAINING")
    print("==============================\n")

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n{DATA_FILE}\n\n"
            "Make sure housing_data.csv is in the same folder "
            "as train_model.py."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(DATA_FILE)

    # Clean column names
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print("Dataset columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Find target column
    # --------------------------------------------------------

    target = None

    for column in PRICE_COLUMNS:

        if column in df.columns:

            target = column
            break

    if target is None:

        raise ValueError(
            "\nPrice column not found.\n"
            f"Expected one of: {PRICE_COLUMNS}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Remove completely empty rows/columns
    # --------------------------------------------------------

    df = df.dropna(
        how="all"
    )

    df = df.dropna(
        axis=1,
        how="all"
    )

    # --------------------------------------------------------
    # Convert target price to numeric
    # --------------------------------------------------------

    df[target] = pd.to_numeric(
        df[target],
        errors="coerce"
    )

    # Remove rows where price is missing
    df = df.dropna(
        subset=[target]
    )

    # --------------------------------------------------------
    # Owner details
    #
    # These should NOT be used for prediction.
    # They are only displayed in the application.
    # --------------------------------------------------------

    display_only_columns = [
        "owner_name",
        "owner_phone",
        "owner_email",

        "Owner_Name",
        "Owner_Phone",
        "Owner_Email"
    ]

    # --------------------------------------------------------
    # Select model features
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns

        if column != target
        and column not in display_only_columns
    ]

    if not feature_columns:

        raise ValueError(
            "No feature columns available for training."
        )

    X = df[feature_columns].copy()

    y = df[target].copy()

    print("\nFeatures used for training:")
    print(feature_columns)

    # ========================================================
    # IMPORTANT:
    # Modern Pandas compatibility
    # ========================================================

    # Explicitly include object, string and category.
    # This removes the Pandas warning you were getting.

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=[
            np.number
        ]
    ).columns.tolist()

    # --------------------------------------------------------
    # Handle any remaining unusual columns
    # --------------------------------------------------------

    remaining_columns = [
        column

        for column in X.columns

        if column not in categorical_columns
        and column not in numerical_columns
    ]

    for column in remaining_columns:

        X[column] = X[column].astype("string")

    categorical_columns.extend(
        remaining_columns
    )

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    # ========================================================
    # PREPROCESSING
    # ========================================================

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # ========================================================
    # RANDOM FOREST MODEL
    # ========================================================

    model = RandomForestRegressor(

        n_estimators=400,

        random_state=42,

        n_jobs=-1,

        min_samples_leaf=1
    )

    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42
    )

    print("\nTraining model...")

    # ========================================================
    # TRAIN
    # ========================================================

    pipeline.fit(
        X_train,
        y_train
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = pipeline.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    metrics = {

        "model": "RandomForestRegressor",

        "r2_score": round(
            float(r2),
            3
        ),

        "mae": round(
            float(mae),
            2
        ),

        "rmse": round(
            float(rmse),
            2
        ),

        "rows": int(
            len(df)
        ),

        "features": feature_columns
    }

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        pipeline,
        MODEL_FILE
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n==============================")
    print("TRAINING COMPLETED")
    print("==============================")

    print("\nMetrics:")
    print(metrics)

    print("\nModel created:")
    print(MODEL_FILE)

    print("\nMetrics file created:")
    print(METRICS_FILE)

    print("\n==============================\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()