from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_moons
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the California housing dataset
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data,
                    columns=california_housing.feature_names)
target = pd.DataFrame(california_housing.target, columns=['MedHouseVal'])

# Selecting a subset of features for demonstration
selected_features = ['MedInc', 'HouseAge',
                     'AveRooms', 'AveBedrms', 'Population']

# Pairplot to visualize relationships between selected features and target
sns.pairplot(data[selected_features].join(target))
plt.show()

# Generating a nonlinearly separable dataset (moons)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

# Plotting the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.title("Nonlinearly Separable Dataset (Moons)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Load the Breast Cancer Wisconsin (Diagnostic) dataset
breast_cancer = load_breast_cancer()

# Create a DataFrame from the dataset
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
target = pd.DataFrame(breast_cancer.target, columns=['diagnosis'])

# Pairplot to visualize relationships between selected features and target
selected_features = ['mean radius', 'mean texture',
                     'mean perimeter', 'mean area', 'mean smoothness']
sns.pairplot(data[selected_features].join(target))
plt.show()

