from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import pandas as pd
import time


class ClassifierEvaluator:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.classifiers = {
            'XGBClassifier': XGBClassifier(),
            'RandomForest': RandomForestClassifier(),
            'KNeighbors': KNeighborsClassifier(),
            'LogisticRegression': LogisticRegression(penalty=None),
            'SVC': SVC(),
            'MLPClassifier': MLPClassifier()
        }
        self.trained_models = {}

    def evaluate(self):
        results_accuracy = []
        results_f1 = []
        results_roc_auc = []
        results_importances = []
        results_train_time = []

        for name, clf in self.classifiers.items():
            start_time = time.time()

            cv_results = cross_validate(clf, self.X_train, self.y_train, scoring=[
                                        'accuracy', 'f1', 'roc_auc'], cv=5)
            results_accuracy.append(cv_results['test_accuracy'].mean())
            results_f1.append(cv_results['test_f1'].mean())
            results_roc_auc.append(cv_results['test_roc_auc'].mean())

            clf.fit(self.X_train, self.y_train)
            end_time = time.time()
            train_time = end_time - start_time
            results_train_time.append(train_time)
            self.trained_models[name] = clf

            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
            else:
                perm_importance = permutation_importance(
                    clf, self.X_test, self.y_test, n_repeats=10, random_state=42, n_jobs=-1)
                importances = perm_importance.importances_mean
            results_importances.append(importances)

        # Create tables
        results_table = pd.DataFrame({
            'Classifier': list(self.classifiers.keys()),
            'Accuracy': results_accuracy,
            'F1 Score': results_f1,
            'ROC AUC': results_roc_auc,
            'Training Time (s)': results_train_time
        })

        importance_table = pd.DataFrame(results_importances, columns=[
                                        'Feature {}'.format(i) for i in range(self.X_train.shape[1])])
        importance_table.insert(0, 'Classifier', list(self.classifiers.keys()))

        return results_table, importance_table, self.trained_models


# Example usage
# Assuming you have your own dataset X and y
X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, random_state=42)

evaluator = ClassifierEvaluator(X, y)
results, importances, trained_models = evaluator.evaluate()

print("Results Table:")
print(results)
print("\nFeature Importances Table:")
print(importances)
print("\nTrained Models:")
print(trained_models)
