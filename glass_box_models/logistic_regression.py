import os
from interpret import show
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
import pandas as pd
set_visualize_provider(InlineProvider())


# Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Target variable (0 for malignant, 1 for benign)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Retrieve the feature names and coefficients
feature_names = data.feature_names
coefficients = model.coef_[0]


# Calculate the exponentiated coefficients
exp_coefficients = np.exp(coefficients)

# Create a DataFrame to display the coefficients and their exponentiated values
coef_table = pd.DataFrame(
    {'Feature': feature_names, 'Coefficient': coefficients, 'Exp(Coefficient)': exp_coefficients})

# Display the table
print(coef_table)

train_data = pd.read_csv(os.path.join(os.getcwd(),
                                      ".\\datasets\\titanic.csv"))
train_data = train_data.dropna()
train_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {
                   'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
train_data = train_data.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X = train_data.drop(columns=['Survived'], axis=1)
Y = train_data['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)
model = LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy*100, '%')
# Retrieve the feature names and coefficients
feature_names = X.columns
coefficients = model.coef_[0]


# Calculate the exponentiated coefficients
exp_coefficients = np.exp(coefficients)

# Create a DataFrame to display the coefficients and their exponentiated values
coef_table = pd.DataFrame(
    {'Feature': feature_names, 'Coefficient': coefficients, 'Exp(Coefficient)': exp_coefficients})

# Display the table
print(coef_table)
