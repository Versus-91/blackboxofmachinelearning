import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Given mean and covariance matrix
mean = [0, 0, 0, 0]
cov = [[1, 0.95, 0, 0], [0.95, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0.5, 1]]

# Number of samples
num_samples = 1000  # Replace with your desired number of samples

# Generate samples
XS = np.random.multivariate_normal(mean, cov, num_samples)

# Extract individual variables
X1 = XS[:, 0]
X2 = XS[:, 1]
X3 = XS[:, 2]
X4 = XS[:, 3]

# Create a DataFrame for seaborn
import pandas as pd

data = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4})

# Show Pearson correlation coefficients
correlation_matrix = data.corr(method='pearson')
print(correlation_matrix)

# Create pairplot
sns.pairplot(data)

# Show Pearson correlation coefficients in a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Pearson Correlation Coefficients')
plt.show()
