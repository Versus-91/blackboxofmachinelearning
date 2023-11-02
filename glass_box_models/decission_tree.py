import pandas as pd
import os
from interpret import show
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from interpret.glassbox import ClassificationTree
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

set_visualize_provider(InlineProvider())

train_data = pd.read_csv(os.path.join(os.getcwd(),
                                      ".\\datasets\\titanic.csv"))
train_data = train_data.dropna()
train_data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {
                   'S': 0, 'C': 1, 'Q': 2}}, inplace=True)
train_data = train_data.drop(
    columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X = train_data.drop(columns=['Survived'], axis=1)
Y = train_data['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)
model = ClassificationTree()
model.fit(X_train, Y_train)
auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
print("AUC: {:.3f}".format(auc))
show(model.explain_global())
