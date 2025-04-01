import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Membaca dataset yang sudah dibersihkan dari outlier
file_path = "HousePricing_no_outliers.csv"  # Gantilah dengan path yang sesuai
df_no_outliers = pd.read_csv(file_path)

# Memilih fitur numerik
numeric_df = df_no_outliers.select_dtypes(include=["number"])

# Sebelum scaling: Visualisasi histogram dari data asli
plt.figure(figsize=(12, 6))
plt.suptitle("Distribusi Data Sebelum Scaling")

# Menyesuaikan jumlah subplot berdasarkan jumlah fitur
num_features = len(numeric_df.columns)
ncols = 5  # Menentukan jumlah kolom
nrows = (num_features // ncols) + 1  # Menentukan jumlah baris berdasarkan jumlah fitur

for i, column in enumerate(numeric_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(numeric_df[column], bins=30, alpha=0.7)
    plt.title(f"Before: {column}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# StandardScaler
scaler_standard = StandardScaler()
scaled_standard = scaler_standard.fit_transform(numeric_df)

# MinMaxScaler
scaler_minmax = MinMaxScaler()
scaled_minmax = scaler_minmax.fit_transform(numeric_df)

# Menyusun dataframe untuk data yang telah discaling
scaled_standard_df = pd.DataFrame(scaled_standard, columns=numeric_df.columns)
scaled_minmax_df = pd.DataFrame(scaled_minmax, columns=numeric_df.columns)

# Setelah scaling: Visualisasi histogram untuk data setelah scaling

plt.figure(figsize=(12, 6))
plt.suptitle("Distribusi Data Setelah Scaling (StandardScaler)")

# StandardScaler
for i, column in enumerate(scaled_standard_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(scaled_standard_df[column], bins=30, alpha=0.7)
    plt.title(f"StandardScaler: {column}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

plt.figure(figsize=(12, 6))
plt.suptitle("Distribusi Data Setelah Scaling (MinMaxScaler)")

# MinMaxScaler
for i, column in enumerate(scaled_minmax_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(scaled_minmax_df[column], bins=30, alpha=0.7)
    plt.title(f"MinMaxScaler: {column}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Output dataframe yang sudah discaling
print("Data Setelah StandardScaler:\n", scaled_standard_df.head())
print("\nData Setelah MinMaxScaler:\n", scaled_minmax_df.head())
