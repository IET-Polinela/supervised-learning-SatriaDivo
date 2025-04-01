import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# === 1. Load Dataset ===
df_outliers = pd.read_csv("HousePricing.csv")  # Dataset dengan outlier
df_clean = pd.read_csv("HousePricing_no_outliers.csv")  # Dataset tanpa outlier

# === 2. Memisahkan Fitur (X) dan Target (y) ===
X_outliers = df_outliers.drop(columns=["SalePrice"])  
y_outliers = df_outliers["SalePrice"]

X_clean = df_clean.drop(columns=["SalePrice"])
y_clean = df_clean["SalePrice"]

# === 3. Mengubah Data Kategorikal ke Numerik ===
X_outliers = pd.get_dummies(X_outliers, drop_first=True)
X_clean = pd.get_dummies(X_clean, drop_first=True)

# Pastikan kedua dataset memiliki kolom yang sama setelah encoding
X_outliers, X_clean = X_outliers.align(X_clean, join='left', axis=1, fill_value=0)

# === 4. Membagi Data Menjadi Training dan Testing Set ===
X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X_outliers, y_outliers, test_size=0.2, random_state=42)
X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# === 5. Scaling Data (StandardScaler) untuk dataset tanpa outlier ===
X_train_clean = X_train_clean.fillna(X_train_clean.mean()) # or X_train_clean.fillna(0)
X_test_clean = X_test_clean.fillna(X_test_clean.mean())  # or X_test_clean.fillna(0)

scaler = StandardScaler()
X_train_clean_scaled = scaler.fit_transform(X_train_clean)
X_test_clean_scaled = scaler.transform(X_test_clean)

# === Impute NaN values in X_train_out (Replace NaN with 0) ===
X_train_out = X_train_out.fillna(0)  # or X_train_out.fillna(X_train_out.mean(), inplace=True)
X_test_out = X_test_out.fillna(0)    # or X_test_out.fillna(X_test_out.mean(), inplace=True)


# === 6. Model Linear Regression ===
model_out = LinearRegression()
model_clean = LinearRegression()

# Melatih Model
model_out.fit(X_train_out, y_train_out)  # Dengan outlier
model_clean.fit(X_train_clean_scaled, y_train_clean)  # Tanpa outlier (scaling diterapkan)

# === 7. Prediksi ===
y_pred_out = model_out.predict(X_test_out)  # Prediksi dengan outlier
y_pred_clean = model_clean.predict(X_test_clean_scaled)  # Prediksi tanpa outlier

# === 8. Evaluasi Model ===
mse_out = mean_squared_error(y_test_out, y_pred_out)
r2_out = r2_score(y_test_out, y_pred_out)

mse_clean = mean_squared_error(y_test_clean, y_pred_clean)
r2_clean = r2_score(y_test_clean, y_pred_clean)

print(f"Model dengan Outlier - MSE: {mse_out:.2f}, R² Score: {r2_out:.2f}")
print(f"Model tanpa Outlier - MSE: {mse_clean:.2f}, R² Score: {r2_clean:.2f}")

# === 9. Visualisasi Scatter Plot Prediksi vs Aktual ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Scatter Plot: Actual vs Predicted")

# Dengan Outlier
sns.scatterplot(x=y_test_out, y=y_pred_out, alpha=0.6, ax=axes[0])
axes[0].plot([y_test_out.min(), y_test_out.max()], [y_test_out.min(), y_test_out.max()], color="red", linestyle="--")
axes[0].set_title("Dengan Outlier")
axes[0].set_xlabel("Actual Values")
axes[0].set_ylabel("Predicted Values")

# Tanpa Outlier
sns.scatterplot(x=y_test_clean, y=y_pred_clean, alpha=0.6, ax=axes[1])
axes[1].plot([y_test_clean.min(), y_test_clean.max()], [y_test_clean.min(), y_test_clean.max()], color="red", linestyle="--")
axes[1].set_title("Tanpa Outlier (Scaling)")
axes[1].set_xlabel("Actual Values")
axes[1].set_ylabel("Predicted Values")

plt.show()

# === 10. Residual Plot ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Residual Plot")

# Dengan Outlier
residuals_out = y_test_out - y_pred_out
sns.scatterplot(x=y_pred_out, y=residuals_out, alpha=0.6, ax=axes[0])
axes[0].axhline(y=0, color="red", linestyle="--")
axes[0].set_title("Dengan Outlier")
axes[0].set_xlabel("Predicted Values")
axes[0].set_ylabel("Residuals")

# Tanpa Outlier
residuals_clean = y_test_clean - y_pred_clean
sns.scatterplot(x=y_pred_clean, y=residuals_clean, alpha=0.6, ax=axes[1])
axes[1].axhline(y=0, color="red", linestyle="--")
axes[1].set_title("Tanpa Outlier (Scaling)")
axes[1].set_xlabel("Predicted Values")
axes[1].set_ylabel("Residuals")

plt.show()

# === 11. Distribusi Residual ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Distribusi Residual")

# Dengan Outlier
sns.histplot(residuals_out, bins=30, kde=True, ax=axes[0])
axes[0].axvline(x=0, color="red", linestyle="--")
axes[0].set_title("Dengan Outlier")
axes[0].set_xlabel("Residuals")

# Tanpa Outlier
sns.histplot(residuals_clean, bins=30, kde=True, ax=axes[1])
axes[1].axvline(x=0, color="red", linestyle="--")
axes[1].set_title("Tanpa Outlier (Scaling)")
axes[1].set_xlabel("Residuals")

plt.show()
