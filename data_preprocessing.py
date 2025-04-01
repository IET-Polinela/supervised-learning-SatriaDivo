import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
file_path = "HousePricing.csv"
df = pd.read_csv(file_path)

# Mengidentifikasi kolom nonnumerik
non_numeric_cols = df.select_dtypes(exclude=["number"]).columns

# Menerapkan encoding untuk fitur-fitur nonnumerik
encoder = LabelEncoder()

for col in non_numeric_cols:
    df[col] = encoder.fit_transform(df[col])

# Menentukan fitur independen (X) dan target/label (Y)
# Misalkan target/label adalah 'SalePrice' (sesuaikan dengan kolom target yang ada pada dataset Anda)
X = df.drop(columns=["SalePrice"])
Y = df["SalePrice"]

# Membagi dataset menjadi training data dan testing data (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Data preprocessing selesai.")
print(f"Dimensi training data: X_train={X_train.shape}, Y_train={Y_train.shape}")
print(f"Dimensi testing data: X_test={X_test.shape}, Y_test={Y_test.shape}")
