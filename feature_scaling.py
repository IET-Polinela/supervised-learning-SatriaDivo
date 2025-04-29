import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Membaca dataset yang sudah dibersihkan dari outlier
file_path = "HousePricing_no_outliers.csv"
df_no_outliers = pd.read_csv(file_path)

# Memilih fitur numerik
numeric_df = df_no_outliers.select_dtypes(include=["number"])

# Drop kolom yang seluruh nilainya NaN agar tidak mengganggu visualisasi dan scaling
valid_numeric_df = numeric_df.dropna(axis=1, how='all')
print(f"Jumlah fitur numerik valid: {valid_numeric_df.shape[1]}")

# Visualisasi sebelum scaling
plt.figure(figsize=(15, 8))
plt.suptitle("Distribusi Data Sebelum Scaling")
ncols = 5
nrows = (len(valid_numeric_df.columns) + ncols - 1) // ncols

for i, col in enumerate(valid_numeric_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(valid_numeric_df[col].dropna(), bins=30, alpha=0.7)
    plt.title(f"Before: {col}")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Distribusi_Sebelum_Scaling.png")  # Menyimpan visualisasi sebelum scaling
plt.show()

# StandardScaler
scaler_standard = StandardScaler()
scaled_standard = scaler_standard.fit_transform(valid_numeric_df)
scaled_standard_df = pd.DataFrame(scaled_standard, columns=valid_numeric_df.columns)

# MinMaxScaler
scaler_minmax = MinMaxScaler()
scaled_minmax = scaler_minmax.fit_transform(valid_numeric_df)
scaled_minmax_df = pd.DataFrame(scaled_minmax, columns=valid_numeric_df.columns)

# Menyimpan hasil scaling
scaled_standard_df.to_csv("HousePricing_StandardScaled.csv", index=False)
scaled_minmax_df.to_csv("HousePricing_MinMaxScaled.csv", index=False)

# Visualisasi setelah StandardScaler
plt.figure(figsize=(15, 8))
plt.suptitle("Distribusi Data Setelah Scaling (StandardScaler)")
for i, col in enumerate(scaled_standard_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(scaled_standard_df[col].dropna(), bins=30, alpha=0.7)
    plt.title(f"Standard: {col}")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("4Distribusi_After_StandardScaler.png")  # Menyimpan visualisasi setelah StandardScaler
plt.show()

# Visualisasi setelah MinMaxScaler
plt.figure(figsize=(15, 8))
plt.suptitle("Distribusi Data Setelah Scaling (MinMaxScaler)")
for i, col in enumerate(scaled_minmax_df.columns, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(scaled_minmax_df[col].dropna(), bins=30, alpha=0.7)
    plt.title(f"MinMax: {col}")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("4Distribusi_After_MinMaxScaler.png")  # Menyimpan visualisasi setelah MinMaxScaler
plt.show()

# Output head hasil scaling
print("Contoh hasil StandardScaler:")
print(scaled_standard_df.head())
print("\nContoh hasil MinMaxScaler:")
print(scaled_minmax_df.head())
