from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes = load_diabetes(as_frame=True)
print(diabetes.head())

X, y = diabetes.data, diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Instantiate Logistic Regression model
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train_normalized, y_train)
print(X_train)

# Instantiate SVM model
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_normalized, y_train)

# Plot partial dependence for a chosen feature index (e.g., feature 2 - body mass index) for Logistic Regression
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(logreg, X_train_normalized,
                                        features=[0, 1], ax=ax, target=0)
plt.title("Partial Dependence Plot - Body Mass Index (Logistic Regression)")
plt.xlabel("Body Mass Index")
plt.ylabel("Partial Dependence")
plt.show()
