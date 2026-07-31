import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "data/cars_features.csv"

PREPROCESSOR_PATH = "models/preprocessor.joblib"

MODEL_PATH = "models/final_car_price_model.joblib"

TARGET = "priceusd"



print("Loading dataset...")

df = pd.read_csv(DATA_PATH)



print("Splitting features and target...")

X = df.drop(columns=[TARGET])
y = df[TARGET]



print("Creating train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



print("Loading preprocessor...")

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)



print("Creating Random Forest model...")


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)



pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])



print("Training model...")

pipeline.fit(
    X_train,
    y_train
)



print("Saving trained model...")


joblib.dump(
    pipeline,
    MODEL_PATH
)



print("Model saved successfully!")

print(f"Saved location: {MODEL_PATH}")