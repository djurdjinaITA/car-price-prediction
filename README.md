# Car Price Prediction

## Project Overview

This project focuses on predicting the price of used cars using machine learning regression models.

The goal is to build a model that estimates the market value of a used vehicle based on its characteristics, such as:

* car brand and model;
* production year;
* mileage;
* fuel type;
* engine volume;
* transmission;
* drive unit;
* vehicle condition;
* car segment.

The target variable is:

```
priceusd
```

The model predicts the estimated price of a used car in US dollars.

---

# Dataset

The dataset used in this project is:

```
cars.csv
```

The dataset contains information about used vehicles.

Original features:

| Column             | Description                     |
| ------------------ | ------------------------------- |
| make               | Car manufacturer                |
| model              | Car model                       |
| priceusd           | Vehicle price (target variable) |
| year               | Production year                 |
| condition          | Vehicle condition               |
| mileage_kilometers | Vehicle mileage                 |
| fuel_type          | Fuel type                       |
| volume_cm3         | Engine volume                   |
| color              | Vehicle color                   |
| transmission       | Transmission type               |
| drive_unit         | Drive system                    |
| segment            | Vehicle segment                 |

Dataset size:

* 56,244 rows
* 12 original columns

---

# Project Structure

```
car-price-prediction/

├── data/
│   ├── cars.csv
│   └── cars_features.csv
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
│
├── models/
│   ├── preprocessor.joblib
│   └── final_car_price_model.joblib
│
├── requirements.txt
└── README.md
```

---

# Exploratory Data Analysis (EDA)

Before training the models, exploratory data analysis was performed in a Jupyter Notebook.

The analysis included:

* checking dataset dimensions;
* inspecting column names and data types;
* displaying random samples;
* separating numerical and categorical features;
* checking missing values;
* identifying possible data quality problems;
* analysing possible outliers.

The goal of EDA was to understand the dataset and define the necessary cleaning and preprocessing steps.

---

# Data Cleaning

Data cleaning was implemented in:

```
src/data_cleaning.py
```

The cleaning process includes:

* standardizing column names;
* removing unnecessary spaces;
* handling missing values;
* converting columns to appropriate data types;
* cleaning categorical values.

---

# Feature Engineering

Feature engineering was implemented in:

```
src/feature_engineering.py
```

New features were created to provide additional information to the model:

### car_age

Represents vehicle age calculated from the production year.

### mileage_per_year

Represents average mileage per year.

### engine_volume_liters

Converts engine volume from cm³ to liters.

### brand_model

Combines manufacturer and model into one feature.

### is_high_mileage

Indicates vehicles with high mileage.

These features help the model better understand factors that influence vehicle prices.

---

# Data Preprocessing

Preprocessing was implemented in:

```
src/data_preprocessing.py
```

The preprocessing pipeline contains:

## Numerical features

* missing value handling using `SimpleImputer`;
* scaling using `StandardScaler`.

## Categorical features

* missing value handling;
* encoding using `OneHotEncoder`.

The preprocessing pipeline was saved as:

```
models/preprocessor.joblib
```

---

# Machine Learning Models

Several regression models were tested:

* Linear Regression;
* Decision Tree Regressor;
* Random Forest Regressor;
* Gradient Boosting Regressor.

All models were trained and evaluated using the same train/test split to ensure fair comparison.

---

# Model Evaluation

The following regression metrics were used:

* MAE (Mean Absolute Error);
* RMSE (Root Mean Squared Error);
* R² score.

MAE was used as the main interpretation metric because it shows the average prediction error in dollars.

Example:

```
MAE = 1000
```

means that the model prediction is wrong by approximately 1000 dollars on average.

---

# Model Comparison Results

| Model             |     MAE |    RMSE |   R² |
| ----------------- | ------: | ------: | ---: |
| Random Forest     | 1029.99 | 2257.64 | 0.92 |
| Decision Tree     | 1316.69 | 2837.51 | 0.87 |
| Gradient Boosting | 1498.12 | 2793.70 | 0.88 |
| Linear Regression | 2043.73 | 4004.45 | 0.74 |

---

# Final Model Selection

Based on the evaluation results, the selected final model is:

```
Random Forest Regressor
```

The model was selected because it achieved:

* the lowest MAE value;
* the highest R² score;
* the best overall prediction performance.

The final trained model is saved as:

```
models/final_car_price_model.joblib
```

---

# How to Run the Project

Install required libraries:

```bash
pip install -r requirements.txt
```

Run data cleaning:

```bash
python src/data_cleaning.py
```

Run feature engineering:

```bash
python src/feature_engineering.py
```

Create preprocessing pipeline:

```bash
python src/data_preprocessing.py
```

Train final model:

```bash
python src/model_training.py
```

Evaluate model:

```bash
python src/model_evaluation.py
```

Compare models:

```bash
python src/model_comparison.py
```

---

# Conclusion

This project demonstrates a complete machine learning workflow for used car price prediction.

The workflow included:

* exploratory data analysis;
* data cleaning;
* feature engineering;
* preprocessing;
* model training;
* model comparison;
* model evaluation.

The final Random Forest model achieved strong performance with:

* R² score: 0.92
* MAE: approximately 1030 USD

The model can provide a useful estimation of used car prices based on vehicle characteristics.
