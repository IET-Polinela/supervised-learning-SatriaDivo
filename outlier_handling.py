import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# === Load Dataset ===
file_path = "HousePricing.csv"
df = pd.read_csv(file_path)

# === Visualisasi Boxplot Awal ===
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(12, 8))
sns.boxplot(data=numeric_df)
plt.xticks(rotation=90)
plt.title('Boxplot Sebelum Menghapus Outlier')
plt.tight_layout()
plt.show()

# === Deteksi Outlier dengan IQR ===
Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)

print(f"Jumlah baris dengan outlier berdasarkan IQR: {outliers_iqr.sum()}")

# Dataset tanpa outlier (IQR)
df_iqr = df[~outliers_iqr]

# === Visualisasi Boxplot Setelah Menghapus Outlier ===
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_iqr.select_dtypes(include=["number"]))
plt.xticks(rotation=90)
plt.title('Boxplot Setelah Menghapus Outlier')
plt.tight_layout()
plt.show()

# === Menyimpan Dataset Tanpa Outlier ===
df_iqr.to_csv("HousePricing_no_outliers.csv", index=False)

print("Dataset tanpa outlier telah disimpan sebagai 'HousePricing_no_outliers.csv'.")
