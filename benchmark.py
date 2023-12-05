import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and start an H2O cluster
h2o.init()

# Convert the training and test data into H2O Frames
train = h2o.H2OFrame(list(X_train) + [y_train.tolist()], column_names=iris.feature_names + ['target'])
test = h2o.H2OFrame(list(X_test) + [y_test.tolist()], column_names=iris.feature_names + ['target'])

# Identify the target column
target_column = 'target'
train[target_column] = train[target_column].asfactor()
test[target_column] = test[target_column].asfactor()

# Set up and train AutoML
aml = H2OAutoML(max_models=10, seed=42)
aml.train(x=iris.feature_names, y=target_column, training_frame=train)

# View the AutoML leaderboard (model performance)
lb = aml.leaderboard
print(lb)

# Get the best model from AutoML
best_model = aml.leader
print(best_model)

# Make predictions on the test set using the best model
predictions = best_model.predict(test)
print(predictions)

# Shutdown the H2O cluster
h2o.shutdown()
