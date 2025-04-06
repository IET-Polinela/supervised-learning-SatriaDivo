import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Dataset yang akan dibandingkan
datasets = {
    "MinMaxScaled": "HousePricing_MinMaxScaled.csv",
    "StandardScaled": "HousePricing_StandardScaled.csv"
}

# Loop untuk setiap dataset
for dataset_name, file_path in datasets.items():
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load dataset
    data = pd.read_csv(file_path)

    # Tangani missing value jika ada
    if data.isnull().values.any():
        print("Terdapat NaN. Mengisi dengan mean...")
        data.fillna(data.mean(), inplace=True)

    # Pisahkan fitur dan target
    X = data[['OverallQual']]
    y = data['SalePrice']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)

    # 2. Polynomial Regression degree=2
    poly2 = PolynomialFeatures(degree=2)
    X_poly2_train = poly2.fit_transform(X_train)
    X_poly2_test = poly2.transform(X_test)

    poly2_model = LinearRegression()
    poly2_model.fit(X_poly2_train, y_train)
    y_pred_poly2 = poly2_model.predict(X_poly2_test)
    mse_poly2 = mean_squared_error(y_test, y_pred_poly2)
    r2_poly2 = r2_score(y_test, y_pred_poly2)

    # 3. Polynomial Regression degree=3
    poly3 = PolynomialFeatures(degree=3)
    X_poly3_train = poly3.fit_transform(X_train)
    X_poly3_test = poly3.transform(X_test)

    poly3_model = LinearRegression()
    poly3_model.fit(X_poly3_train, y_train)
    y_pred_poly3 = poly3_model.predict(X_poly3_test)
    mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
    r2_poly3 = r2_score(y_test, y_pred_poly3)

    # Cetak hasil evaluasi
    print(f"Linear Regression - MSE: {mse_linear:.2f}, R2: {r2_linear:.2f}")
    print(f"Polynomial Regression (Degree 2) - MSE: {mse_poly2:.2f}, R2: {r2_poly2:.2f}")
    print(f"Polynomial Regression (Degree 3) - MSE: {mse_poly3:.2f}, R2: {r2_poly3:.2f}")

    # Visualisasi
    plt.figure(figsize=(15, 4))
    plt.suptitle(f"Model Comparison for {dataset_name}", fontsize=16)

    # Linear
    plt.subplot(1, 3, 1)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred_linear, color='red', label='Predicted')
    plt.title("Linear Regression")
    plt.xlabel("Id")
    plt.ylabel("SalePrice")
    plt.legend()

    # Poly 2
    plt.subplot(1, 3, 2)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred_poly2, color='green', label='Predicted')
    plt.title("Polynomial Regression (Degree 2)")
    plt.xlabel("Id")
    plt.legend()

    # Poly 3
    plt.subplot(1, 3, 3)
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred_poly3, color='purple', label='Predicted')
    plt.title("Polynomial Regression (Degree 3)")
    plt.xlabel("Id")
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
