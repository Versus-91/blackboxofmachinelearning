# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
print(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train )

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy and print a classification report
accuracy = accuracy_score(y_test , y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

