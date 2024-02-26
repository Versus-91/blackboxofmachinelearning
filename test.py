import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier

# Load the wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importances for each tree
importances = np.array([tree.feature_importances_ for tree in rf_model.estimators_])

# Calculate mean and standard deviation of feature importances
mean_importances = np.mean(importances, axis=0)
std_importances = np.std(importances, axis=0)

# Sort features by their importance
sorted_indices = np.argsort(mean_importances)[::-1]
sorted_feature_names = [wine.feature_names[i] for i in sorted_indices]
sorted_mean_importances = mean_importances[sorted_indices]
sorted_std_importances = std_importances[sorted_indices]

# Plot horizontal bar chart with error bars
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_mean_importances, xerr=sorted_std_importances, color='skyblue', alpha=0.6)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance with Standard Deviation')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()
