import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Path to your dataset file
file_path = 'HousePricing_no_outliers.csv'  

# Load the dataset
data = pd.read_csv(file_path)

X = data[['Id']]  
y = data['SalePrice']     

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predicting with Linear Regression
y_pred_linear = linear_regressor.predict(X_test)

# Calculate MSE and R2 for Linear Regression
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# 2. Polynomial Regression (Degree 2)
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X_train)

poly2_regressor = LinearRegression()
poly2_regressor.fit(X_poly2, y_train)

# Predicting with Polynomial Regression (Degree 2)
X_test_poly2 = poly2.transform(X_test)
y_pred_poly2 = poly2_regressor.predict(X_test_poly2)

# Calculate MSE and R2 for Polynomial Regression (Degree 2)
mse_poly2 = mean_squared_error(y_test, y_pred_poly2)
r2_poly2 = r2_score(y_test, y_pred_poly2)

# 3. Polynomial Regression (Degree 3)
poly3 = PolynomialFeatures(degree=3)
X_poly3 = poly3.fit_transform(X_train)

poly3_regressor = LinearRegression()
poly3_regressor.fit(X_poly3, y_train)

# Predicting with Polynomial Regression (Degree 3)
X_test_poly3 = poly3.transform(X_test)
y_pred_poly3 = poly3_regressor.predict(X_test_poly3)

# Calculate MSE and R2 for Polynomial Regression (Degree 3)
mse_poly3 = mean_squared_error(y_test, y_pred_poly3)
r2_poly3 = r2_score(y_test, y_pred_poly3)

# 4. Visualizations
plt.figure(figsize=(12, 6))

# Plotting Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred_linear, color='red')
plt.title('Linear Regression')

# Plotting Polynomial Regression (Degree 2)
plt.subplot(1, 3, 2)
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred_poly2, color='green')
plt.title('Polynomial Regression (Degree 2)')

# Plotting Polynomial Regression (Degree 3)
plt.subplot(1, 3, 3)
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred_poly3, color='purple')
plt.title('Polynomial Regression (Degree 3)')

plt.tight_layout()
plt.show()

# Output the evaluation results
print(f"Linear Regression - MSE: {mse_linear}, R2: {r2_linear}")
print(f"Polynomial Regression (Degree 2) - MSE: {mse_poly2}, R2: {r2_poly2}")
print(f"Polynomial Regression (Degree 3) - MSE: {mse_poly3}, R2: {r2_poly3}")
