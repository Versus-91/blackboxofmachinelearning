import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 200
n_features = 10

# Generate features randomly from a uniform distribution
X = np.random.rand(n_samples, n_features)

# Define the true model
def true_model(X):
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5)**2 + 10 * X[:, 3] + 5 * X[:, 4]

# Generate target variable based on the true model plus Gaussian noise
noise = np.random.normal(loc=0, scale=1.0, size=n_samples)
y = true_model(X) + noise

# Split data into training and testing sets
X_train, X_test = X[:100], X[100:]
y_train, y_test = y[:100], y[100:]
import matplotlib.pyplot as plt

# Function to plot the effect of a single variable on y while fixing others at median values
def plot_variable_effect(X, y, variable_index, variable_name):
    median_values = np.median(X, axis=0)
    fixed_variables = np.tile(median_values, (X.shape[0], 1))
    varying_variable = np.linspace(np.min(X[:, variable_index]), np.max(X[:, variable_index]), num=100)

    X_varying = fixed_variables.copy()
    X_varying[:, variable_index] = varying_variable

    y_pred = true_model(X_varying)

    plt.plot(X_varying[:, variable_index], y_pred, label=variable_name)
    plt.xlabel(variable_name)
    plt.ylabel('y')
    plt.title(f'Effect of {variable_name} on y')
    plt.legend()
    plt.show()

# Plot the effect of each variable on y
for i in range(n_features):
    plot_variable_effect(X_train, y_train, i, f'x{i+1}')
