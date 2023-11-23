from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Fit Logistic Regression model
logistic = LogisticRegression(max_iter=1000)
logistic.fit(X, y)

# Fit SVM model
svm = SVC(kernel='rbf', probability=True)
svm.fit(X, y)

# Plot partial dependence for 'petal width (cm)' feature for Logistic Regression
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    logistic, X, features = [0, 1, 2], ax=ax, target=0)
plt.title("Partial Dependence Plot - Petal Width (Logistic Regression)")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Partial Dependence")
plt.show()
