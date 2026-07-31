import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression


DATA_PATH = "data/cars_features.csv"
MODEL_PATH = "models/car_price_model.joblib"

TARGET = "priceusd"


def create_pipeline():

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


    model = LinearRegression()


    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


    return pipeline, X, y



def main():

    print("Loading data...")


    pipeline, X, y = create_pipeline()


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    print("Training model...")


    pipeline.fit(
        X_train,
        y_train
    )


    joblib.dump(
        pipeline,
        MODEL_PATH
    )


    print("Model saved successfully!")
    print(MODEL_PATH)



if __name__ == "__main__":
    main()