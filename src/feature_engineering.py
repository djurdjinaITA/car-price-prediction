import pandas as pd


INPUT_PATH = "data/cars_cleaned.csv"
OUTPUT_PATH = "data/cars_features.csv"

CURRENT_YEAR = 2026


# Create vehicle age feature
def _add_car_age(df):
    df = df.copy()

    df["car_age"] = CURRENT_YEAR - df["year"]

    return df


# Create average mileage per year feature
def _add_mileage_per_year(df):
    df = df.copy()

    df["mileage_per_year"] = (
        df["mileage_kilometers"] /
        df["car_age"].replace(0, 1)
    )

    return df


# Convert engine volume from cm3 to liters
def _add_engine_volume_liters(df):
    df = df.copy()

    df["engine_volume_liters"] = (
        df["volume_cm3"] / 1000
    )

    return df


# Create high mileage indicator
def _add_high_mileage(df):
    df = df.copy()

    df["is_high_mileage"] = (
        df["mileage_kilometers"] > 200000
    ).astype(int)

    return df


# Combine make and model
def _add_brand_model(df):
    df = df.copy()

    df["brand_model"] = (
        df["make"] + "_" + df["model"]
    )

    return df


# Full feature engineering pipeline
def create_features(df):

    df_features = (
        df
        .pipe(_add_car_age)
        .pipe(_add_mileage_per_year)
        .pipe(_add_engine_volume_liters)
        .pipe(_add_high_mileage)
        .pipe(_add_brand_model)
        .reset_index(drop=True)
    )

    return df_features



def main():

    print("Loading cleaned dataset...")

    df = pd.read_csv(INPUT_PATH)


    print("Creating new features...")

    df_features = create_features(df)


    print("Saving feature dataset...")

    df_features.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("Finished!")
    print(df_features.shape)



if __name__ == "__main__":
    main()