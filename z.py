
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Create a DataFrame from the iris dataset
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

# Convert numeric species values to species names
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Create pairplot using Seaborn
sns.pairplot(iris_df, hue='species', diag_kind='hist')
