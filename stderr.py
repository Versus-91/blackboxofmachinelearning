import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Initialize logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_probs = logreg.predict_proba(X_test)

# Create a violin plot to visualize predicted probabilities
plt.figure(figsize=(8, 6))
sns.violinplot(data=y_probs, inner="points")
plt.xlabel('Classes')
plt.ylabel('Predicted Probability')
plt.title('Violin Plot of Predicted Probabilities for Iris Classes')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names)
plt.show()
