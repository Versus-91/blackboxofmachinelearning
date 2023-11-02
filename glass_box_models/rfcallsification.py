import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X, y = data.data, data.target

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)

rf_classifier.fit(X, y)

importances = rf_classifier.feature_importances_

feature_names = data.feature_names

indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature {feature_names[indices[f]]}: {importances[indices[f]]}")

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=45)
plt.show()
