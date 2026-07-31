import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split


DATA_PATH = "data/cars_features.csv"
MODEL_PATH = "models/car_price_model.joblib"

TARGET = "priceusd"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


print("Splitting features and target...")

X = df.drop(columns=[TARGET])
y = df[TARGET]


print("Creating the same train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Loading trained model...")

model = joblib.load(MODEL_PATH)


print("Making predictions...")

y_pred = model.predict(X_test)



print("Calculating regression metrics...")


mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)



metrics = pd.DataFrame({
    "metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ],
    "value": [
        mae,
        mse,
        rmse,
        r2
    ]
})


# Format numbers for easier reading
metrics["value"] = metrics["value"].round(2)


print("\nRegression metrics:")
print(metrics)



print("\nCreating prediction analysis table...")


prediction_analysis = pd.DataFrame({
    "actual_price": y_test.values,
    "predicted_price": y_pred
})


prediction_analysis["error"] = (
    prediction_analysis["actual_price"]
    - prediction_analysis["predicted_price"]
)


prediction_analysis["absolute_error"] = (
    prediction_analysis["error"]
    .abs()
)



print("\nPrediction examples:")

print(
    prediction_analysis.sample(
        10,
        random_state=42
    )
)



print("\nLargest prediction errors:")

print(
    prediction_analysis
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(10)
)