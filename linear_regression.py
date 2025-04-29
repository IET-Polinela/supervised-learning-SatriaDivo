import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# === 1. Load Dataset ===
df_outliers = pd.read_csv("HousePricing.csv")
df_minmax = pd.read_csv("HousePricing_MinMaxScaled.csv")
df_standard = pd.read_csv("HousePricing_StandardScaled.csv")

# === 2. Memisahkan Fitur (X) dan Target (y) ===
def prepare_data(df):
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Menangani missing values pada fitur numerik dan kategorikal
    imputer_num = SimpleImputer(strategy='mean')  # Mengganti NaN dengan mean
    imputer_cat = SimpleImputer(strategy='most_frequent')  # Mengganti NaN dengan modus

    # Pisahkan fitur numerik dan kategorikal
    num_features = X.select_dtypes(include=[np.number]).columns
    cat_features = X.select_dtypes(include=['object']).columns

    # Imputasi nilai NaN hanya jika ada fitur dengan tipe data tersebut
    if len(num_features) > 0:
        X[num_features] = imputer_num.fit_transform(X[num_features])

    if len(cat_features) > 0:
        X[cat_features] = imputer_cat.fit_transform(X[cat_features])

    # Encoding fitur kategorikal jika ada
    if len(cat_features) > 0:
        X = pd.get_dummies(X, drop_first=True, dummy_na=False)

    # Debugging: Periksa apakah masih ada NaN
    print(f"Jumlah NaN setelah imputasi: {X.isnull().sum().sum()}")  # Harus 0

    return X, y

X_outliers, y_outliers = prepare_data(df_outliers)
X_minmax, y_minmax = prepare_data(df_minmax)
X_standard, y_standard = prepare_data(df_standard)

# Pastikan semua dataset memiliki kolom yang sama setelah encoding
X_outliers, X_minmax = X_outliers.align(X_minmax, join='left', axis=1, fill_value=0)
X_outliers, X_standard = X_outliers.align(X_standard, join='left', axis=1, fill_value=0)

# Debugging: Pastikan tidak ada NaN setelah imputasi dan encoding
assert not X_outliers.isnull().values.any(), "X_outliers masih memiliki NaN!"
assert not X_minmax.isnull().values.any(), "X_minmax masih memiliki NaN!"
assert not X_standard.isnull().values.any(), "X_standard masih memiliki NaN!"

# === 3. Membagi Data Menjadi Training dan Testing Set ===
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outliers, y_outliers, test_size=0.2, random_state=42)
X_train_minmax, X_test_minmax, y_train_minmax, y_test_minmax = train_test_split(X_minmax, y_minmax, test_size=0.2, random_state=42)
X_train_standard, X_test_standard, y_train_standard, y_test_standard = train_test_split(X_standard, y_standard, test_size=0.2, random_state=42)

# === 4. Melatih Model Linear Regression ===
model_out = LinearRegression()
model_minmax = LinearRegression()
model_standard = LinearRegression()

model_out.fit(X_train_out, y_train_out)
model_minmax.fit(X_train_minmax, y_train_minmax)
model_standard.fit(X_train_standard, y_train_standard)

# === 5. Prediksi ===
y_pred_out = model_out.predict(X_test_out)
y_pred_minmax = model_minmax.predict(X_test_minmax)
y_pred_standard = model_standard.predict(X_test_standard)

# === 6. Evaluasi Model ===
mse_out = mean_squared_error(y_test_out, y_pred_out)
r2_out = r2_score(y_test_out, y_pred_out)

mse_minmax = mean_squared_error(y_test_minmax, y_pred_minmax)
r2_minmax = r2_score(y_test_minmax, y_pred_minmax)

mse_standard = mean_squared_error(y_test_standard, y_pred_standard)
r2_standard = r2_score(y_test_standard, y_pred_standard)

print(f"Model dengan Outlier - MSE: {mse_out:.2f}, R² Score: {r2_out:.2f}")
print(f"Model MinMaxScaler - MSE: {mse_minmax:.2f}, R² Score: {r2_minmax:.2f}")
print(f"Model StandardScaler - MSE: {mse_standard:.2f}, R² Score: {r2_standard:.2f}")

# === 7. Visualisasi Scatter Plot Prediksi vs Aktual ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Scatter Plot: Actual vs Predicted")

sns.scatterplot(x=y_test_out, y=y_pred_out, alpha=0.6, ax=axes[0])
axes[0].plot([y_test_out.min(), y_test_out.max()], [y_test_out.min(), y_test_out.max()], color="red", linestyle="--")
axes[0].set_title("Dengan Outlier")
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Predicted Values")

sns.scatterplot(x=y_test_minmax, y=y_pred_minmax, alpha=0.6, ax=axes[1])
axes[1].plot([y_test_minmax.min(), y_test_minmax.max()], [y_test_minmax.min(), y_test_minmax.max()], color="red", linestyle="--")
axes[1].set_title("MinMaxScaler")
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")

sns.scatterplot(x=y_test_standard, y=y_pred_standard, alpha=0.6, ax=axes[2])
axes[2].plot([y_test_standard.min(), y_test_standard.max()], [y_test_standard.min(), y_test_standard.max()], color="red", linestyle="--")
axes[2].set_title("StandardScaler")
axes[2].set_xlabel("Actual Values")
axes[2].set_ylabel("Predicted Values")

plt.savefig("5scatter_plot.png")  # Menyimpan visualisasi scatter plot
plt.show()

# === 8. Visualisasi Residual Plot ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Residual Plot")

residuals_out = y_test_out - y_pred_out
sns.scatterplot(x=y_pred_out, y=residuals_out, alpha=0.6, ax=axes[0])
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_title("Dengan Outlier")
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("Residuals")

residuals_minmax = y_test_minmax - y_pred_minmax
sns.scatterplot(x=y_pred_minmax, y=residuals_minmax, alpha=0.6, ax=axes[1])
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_title("MinMaxScaler")
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals")

residuals_standard = y_test_standard - y_pred_standard
sns.scatterplot(x=y_pred_standard, y=residuals_standard, alpha=0.6, ax=axes[2])
axes[2].axhline(y=0, color="red", linestyle="--")
axes[2].set_title("StandardScaler")
axes[2].set_xlabel("Predicted Values")
axes[2].set_ylabel("Residuals")

plt.savefig("5residual_plot.png")  # Menyimpan visualisasi residual plot
plt.show()

# === 9. Visualisasi Distribusi Residual ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Distribusi Residual")

sns.histplot(residuals_out, bins=30, kde=True, ax=axes[0])
axes[0].set_title("Dengan Outlier")

sns.histplot(residuals_minmax, bins=30, kde=True, ax=axes[1])
axes[1].set_title("MinMaxScaler")

sns.histplot(residuals_standard, bins=30, kde=True, ax=axes[2])
axes[2].set_title("StandardScaler")

plt.savefig("5residual_distribution.png")  # Menyimpan visualisasi distribusi residual
plt.show()
