import pandas as pd

# Load dataset
file_path = "HousePricing.csv"
df = pd.read_csv(file_path)

# Menghitung statistik deskriptif untuk semua fitur
stats = df.describe().T
stats["median"] = df.median()

# Menampilkan statistik yang diminta
stats = stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
print("Statistik Deskriptif:\n", stats)

# Menampilkan jumlah nilai yang hilang per kolom
missing_values = df.isnull().sum()
print("\nJumlah Nilai yang Hilang:\n", missing_values[missing_values > 0])

# Mengisi nilai yang hilang pada kolom LotFrontage dengan median
if 'LotFrontage' in df.columns:
    df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    print("\nNilai yang hilang pada LotFrontage telah diisi dengan median.")

# Menampilkan kembali jumlah nilai yang hilang setelah imputasi
missing_values_after = df.isnull().sum()
print("\nJumlah Nilai yang Hilang Setelah Imputasi:\n", missing_values_after[missing_values_after > 0])
