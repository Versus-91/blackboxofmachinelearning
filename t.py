from dython.nominal import associations  # correlation calculation
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml

boston = fetch_openml(data_id=43465)
# Load Boston Housing dataset
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['target'] = boston.target

# Split the dataset into features and target variable
X = boston.data  # Features
y = boston.target  # Target variable (House prices)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
# Retrieve feature importance (coefficients)
feature_importance = pd.DataFrame({
    'Feature': boston.feature_names,
    'Coefficient': model.coef_
})

# Sort features by their absolute coefficient values
feature_importance['Abs_Coefficient'] = np.abs(
    feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(
    by='Abs_Coefficient', ascending=False)
# Plotting the coefficients
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.title('Feature Importances - Coefficients of Linear Regression Model')
plt.grid(axis='x')
plt.show()
# Compute only default value is False, it give correlation heatmap of all variable.setdefault()fig, ax = plt.subplots(figsize = (12, 10))
correlation_matrix = associations(
    data, mark_columns=True, compute_only=False, figsize=(15, 15),annot=True)

