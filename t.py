from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, shuffle=False)

# Define different lambda values (regularization strengths)
lambda_values = [0.1, 1, 10]

# Train a logistic regression model for each lambda value
for lambda_val in lambda_values:
    # Initialize logistic regression model with specified lambda
    logreg = LogisticRegression(
        C=100, solver='liblinear', multi_class='auto')

    # Fit the model
    logreg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = logreg.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Print lambda value and corresponding accuracy
    print(f"Lambda: {lambda_val}, Accuracy: {accuracy:.4f}")
