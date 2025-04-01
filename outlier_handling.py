import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
file_path = "HousePricing.csv"
df = pd.read_csv(file_path)

# Menampilkan boxplot untuk setiap fitur numerik untuk visualisasi outlier
numeric_df = df.select_dtypes(include=["number"])

# Membuat boxplot untuk semua fitur numerik
plt.figure(figsize=(12, 8))
sns.boxplot(data=numeric_df)
plt.xticks(rotation=90)
plt.title('Boxplot untuk Semua Fitur Numerik')
plt.show()

# Mengidentifikasi outlier menggunakan Z-Score
z_scores = np.abs(stats.zscore(numeric_df))
outliers_zscore = (z_scores > 3).all(axis=1)

# Menampilkan jumlah data dengan outlier (menggunakan Z-Score)
print(f"Jumlah baris dengan outlier berdasarkan Z-Score: {sum(outliers_zscore)}")

# Menghapus outlier (menggunakan Z-Score)
df_no_outliers = df[~outliers_zscore]

# Visualisasi boxplot setelah menghapus outlier
numeric_df_no_outliers = df_no_outliers.select_dtypes(include=["number"])
plt.figure(figsize=(12, 8))
sns.boxplot(data=numeric_df_no_outliers)
plt.xticks(rotation=90)
plt.title('Boxplot Setelah Menghapus Outlier')
plt.show()

# Menyimpan dataset dengan dan tanpa outlier
df_no_outliers.to_csv("HousePricing_no_outliers.csv", index=False)

print("Dataset dengan outlier telah dihapus dan disimpan sebagai 'HousePricing_no_outliers.csv'.")
