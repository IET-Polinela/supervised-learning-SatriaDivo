import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# === Load Dataset ===
df_minmax = pd.read_csv("HousePricing_MinMaxScaled.csv")
df_standard = pd.read_csv("HousePricing_StandardScaled.csv")
df_no_outliers = pd.read_csv("HousePricing_no_outliers.csv")

# Fungsi preprocessing
def prepare_data(df):
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    y = df["SalePrice"]

    # Ubah kolom kategorikal ke dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Hapus kolom yang semua nilainya NaN
    X = X.dropna(axis=1, how='all')

    # Imputasi nilai kosong dengan mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Buat DataFrame baru dengan kolom yang cocok
    X = pd.DataFrame(X_imputed, columns=X.columns)

    return X, y

# Siapkan dataset
datasets = {
    "MinMaxScaler": prepare_data(df_minmax),
    "StandardScaler": prepare_data(df_standard),
    "No Outliers": prepare_data(df_no_outliers)
}

# Nilai K untuk KNN
k_values = [3, 5, 7]

# Simpan hasil perbandingan
comparison_results = {}

# Loop untuk masing-masing dataset
for name, (X, y) in datasets.items():
    print(f"\n=== Dataset: {name} ===")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    result = {}

    # --- Linear Regression ---
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    result["Linear Regression"] = {
        "MSE": mean_squared_error(y_test, y_pred_lr),
        "R²": r2_score(y_test, y_pred_lr)
    }

    # --- Polynomial Regression degree = 2 ---
    poly2 = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly2.fit(X_train, y_train)
    y_pred_poly2 = poly2.predict(X_test)
    result["Polynomial (deg=2)"] = {
        "MSE": mean_squared_error(y_test, y_pred_poly2),
        "R²": r2_score(y_test, y_pred_poly2)
    }

    # --- Polynomial Regression degree = 3 ---
    poly3 = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    poly3.fit(X_train, y_train)
    y_pred_poly3 = poly3.predict(X_test)
    result["Polynomial (deg=3)"] = {
        "MSE": mean_squared_error(y_test, y_pred_poly3),
        "R²": r2_score(y_test, y_pred_poly3)
    }

    # --- KNN Regression ---
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        result[f"KNN (k={k})"] = {
            "MSE": mean_squared_error(y_test, y_pred_knn),
            "R²": r2_score(y_test, y_pred_knn)
        }

    # Simpan hasil
    comparison_results[name] = result

# === Cetak Nilai dan Visualisasi ===
for name in comparison_results:
    print(f"\n=== Hasil Evaluasi Model: {name} Dataset ===")
    print("{:<20s} | {:>12s} | {:>6s}".format("Model", "MSE", "R²"))
    print("-" * 45)

    model_names = list(comparison_results[name].keys())
    mse_values = []
    r2_values = []

    for model in model_names:
        mse = comparison_results[name][model]["MSE"]
        r2 = comparison_results[name][model]["R²"]
        mse_values.append(mse)
        r2_values.append(r2)
        print("{:<20s} | {:12.2f} | {:6.3f}".format(model, mse, r2))

    # Visualisasi
    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Perbandingan Model pada {name} Dataset", fontsize=14)

    # MSE plot
    plt.subplot(1, 2, 1)
    plt.barh(model_names, mse_values, color='skyblue')
    plt.xlabel("MSE")
    plt.title("Perbandingan MSE")

    # R² plot
    plt.subplot(1, 2, 2)
    plt.barh(model_names, r2_values, color='salmon')
    plt.xlabel("R² Score")
    plt.title("Perbandingan R²")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
