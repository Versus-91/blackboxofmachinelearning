import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Create a Random Forest Classifier
random_forestr_classifier = RandomForestClassifier(
    n_estimators=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Train the model
random_forestr_classifier.fit(X_train, y_train)
random_forestr_classifier.predict(X_test)
model_feature_importancs = random_forestr_classifier.feature_importances_
std = np.std(
    [tree.feature_importances_ for tree in random_forestr_classifier.estimators_], axis=0)
feature_names = data.feature_names
print(feature_names)
forest_importances = pd.Series(model_feature_importancs, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(10, 8))
PartialDependenceDisplay.from_estimator(mc_clf, X, features)
