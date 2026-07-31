import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


DATA_PATH = "data/cars_features.csv"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

TARGET = "priceusd"


def create_preprocessor():

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]


    numeric_features = [
        "year",
        "mileage_kilometers",
        "volume_cm3",
        "car_age",
        "mileage_per_year",
        "engine_volume_liters",
        "is_high_mileage"
    ]


    categorical_features = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment",
        "brand_model"
    ]


    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])


    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])


    return preprocessor, X, y



def main():

    print("Creating preprocessing pipeline...")

    preprocessor, X, y = create_preprocessor()

    preprocessor.fit(X)

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print("Preprocessor saved successfully!")
    print(f"Target variable: {TARGET}")
    print(f"Number of features: {X.shape[1]}")



if __name__ == "__main__":
    main()