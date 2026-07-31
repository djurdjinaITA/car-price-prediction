import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_PATH = "data/cars_features.csv"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

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



models = {

    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    )
}



results = []



for name, model in models.items():

    print(f"Training {name}...")


    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    mse = mean_squared_error(
        y_test,
        predictions
    )


    rmse = mse ** 0.5


    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })



results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="MAE"
)


results_df[
    ["MAE", "RMSE", "R2"]
] = results_df[
    ["MAE", "RMSE", "R2"]
].round(2)



print("\nModel comparison:")
print(results_df)