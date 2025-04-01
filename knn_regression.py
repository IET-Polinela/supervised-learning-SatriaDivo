import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder  # Import OneHotEncoder
from sklearn.impute import SimpleImputer # Import SimpleImputer

# Path to your dataset file
file_path = 'HousePricing_no_outliers.csv'

# Load the dataset
data = pd.read_csv(file_path)

# ---->  Convert categorical features to numerical using OneHotEncoder:
categorical_features = data.select_dtypes(include=['object']).columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # sparse=False for numpy array
encoded_data = encoder.fit_transform(data[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

# Concatenate encoded features with numerical features
numerical_features = data.select_dtypes(exclude=['object']).columns
final_data = pd.concat([data[numerical_features], encoded_df], axis=1)


# Split the dataset into features (X) and target (y)
X = final_data.drop(['SalePrice', 'Id'], axis=1)  # Features (Exclude 'SalePrice' and 'Id') # Update this line
y = final_data['SalePrice']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation to handle NaN values before scaling
imputer = SimpleImputer(strategy='mean') # or strategy='median', 'most_frequent', 'constant'
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Feature Scaling (Standardization) for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Regression with K = 3, 5, and 7
k_values = [3, 5, 7]
results = {}

for k in k_values:
   # Initialize the KNN regressor
   knn = KNeighborsRegressor(n_neighbors=k)
   
   # Train the model
   knn.fit(X_train_scaled, y_train)
   
   # Predict on the test set
   y_pred = knn.predict(X_test_scaled)
   
   # Calculate MSE and R²
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   
   # Store the results
   results[k] = {'MSE': mse, 'R²': r2}
   
   # Print the results
   print(f"KNN Regression (K = {k}):")
   print(f"MSE: {mse}")
   print(f"R²: {r2}")
   print("-" * 40)

# Visualize comparison of MSE and R² for KNN with K=3,5,7
k_values_list = list(results.keys())
mse_values = [results[k]['MSE'] for k in k_values_list]
r2_values = [results[k]['R²'] for k in k_values_list]

plt.figure(figsize=(12, 6))

# Plotting MSE for KNN
plt.subplot(1, 2, 1)
plt.plot(k_values_list, mse_values, marker='o', linestyle='-', color='b')
plt.title('KNN Regression: MSE Comparison')
plt.xlabel('K (Neighbors)')
plt.ylabel('MSE')

# Plotting R² for KNN
plt.subplot(1, 2, 2)
plt.plot(k_values_list, r2_values, marker='o', linestyle='-', color='r')
plt.title('KNN Regression: R² Comparison')
plt.xlabel('K (Neighbors)')
plt.ylabel('R²')

plt.tight_layout()
plt.show()

# Comparing KNN with Linear and Polynomial Regression
# Assuming you already have results from Linear and Polynomial Regression saved
linear_results = {'MSE': 7671441339.93, 'R²': -0.000145}
poly_2_results = {'MSE': 7673256012.83, 'R²': -0.000381}
poly_3_results = {'MSE': 7685831610.81, 'R²': -0.002021}

# Comparison data
models = ['Linear Regression', 'Poly Regression (Degree 2)', 'Poly Regression (Degree 3)', 'KNN (K=3)', 'KNN (K=5)', 'KNN (K=7)']
mse_comparison = [linear_results['MSE'], poly_2_results['MSE'], poly_3_results['MSE'], results[3]['MSE'], results[5]['MSE'], results[7]['MSE']]
r2_comparison = [linear_results['R²'], poly_2_results['R²'], poly_3_results['R²'], results[3]['R²'], results[5]['R²'], results[7]['R²']]

# Visualizing MSE and R² Comparison
plt.figure(figsize=(12, 6))

# MSE comparison plot
plt.subplot(1, 2, 1)
plt.barh(models, mse_comparison, color='b')
plt.title('Model Comparison: MSE')
plt.xlabel('MSE')

# R² comparison plot
plt.subplot(1, 2, 2)
plt.barh(models, r2_comparison, color='r')
plt.title('Model Comparison: R²')
plt.xlabel('R²')

plt.tight_layout()
plt.show()
