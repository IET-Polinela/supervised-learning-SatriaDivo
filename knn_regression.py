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
    "StandardScaler": prepare_data(df_standard)
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
    print(f"\n=== Hasil Evaluasi Model: {name} Dataset ===")
    print("{:<20s} | {:>12s} | {:>6s}".format("Model", "MSE", "R²"))
    print("-" * 45)

    model_names = list(result.keys())
    mse_values = []
    r2_values = []

    for model in model_names:
        mse = result[model]["MSE"]
        r2 = result[model]["R²"]
        mse_values.append(mse)
        r2_values.append(r2)
        print("{:<20s} | {:12.2f} | {:6.3f}".format(model, mse, r2))

    # Visualisasi Bar Chart
    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Perbandingan Model pada {name} Dataset", fontsize=14)

    # MSE
    plt.subplot(1, 2, 1)
    plt.barh(model_names, mse_values, color='skyblue')
    plt.xlabel("MSE")
    plt.title("Perbandingan MSE")

    # R²
    plt.subplot(1, 2, 2)
    plt.barh(model_names, r2_values, color='salmon')
    plt.xlabel("R² Score")
    plt.title("Perbandingan R²")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"7bar_chart_{name}.png")  # Menyimpan bar chart
    plt.show()

    # === SCATTER PLOT Prediksi vs Aktual ===

    # KNN Grouped Plot
    plt.figure(figsize=(6, 6))
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.5, label=f"KNN k={k}")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Aktual vs Prediksi - KNN ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"7knn_scatter_{name}.png")  # Menyimpan KNN scatter plot
    plt.show()

    # Polynomial Regression Grouped Plot
    plt.figure(figsize=(6, 6))
    y_pred_poly2 = poly2.predict(X_test)
    y_pred_poly3 = poly3.predict(X_test)
    plt.scatter(y_test, y_pred_poly2, alpha=0.5, label="Poly deg=2")
    plt.scatter(y_test, y_pred_poly3, alpha=0.5, label="Poly deg=3")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Aktual vs Prediksi - Polynomial ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"7poly_scatter_{name}.png")  # Menyimpan Polynomial scatter plot
    plt.show()

    # Linear Regression Plot
    plt.figure(figsize=(6, 6))
    y_pred_lr = lr.predict(X_test)
    plt.scatter(y_test, y_pred_lr, alpha=0.5, label="Linear Regression", color="orange")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"Aktual vs Prediksi - Linear Regression ({name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"7lr_scatter_{name}.png")  # Menyimpan Linear Regression scatter plot
    plt.show()
