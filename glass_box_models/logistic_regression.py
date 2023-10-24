from interpret import show
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

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

feature_names = data.feature_names
coefficients = model.coef_[0]

# Calculate the exponentiated coefficients
exp_coefficients = np.exp(coefficients)

# Create a DataFrame to display the coefficients and their exponentiated values
coef_table = pd.DataFrame(
    {'Feature': feature_names, 'Coefficient': coefficients, 'Exp(Coefficient)': exp_coefficients})

# Display the table
print(coef_table)
