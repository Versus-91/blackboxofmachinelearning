from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost


df = pd.read_csv("./datasets/heart.csv")
df.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg_type', 'max_heart_rate_achieved',
              'exercise_induced_angina', 'st_depression', 'st_slope_type', 'num_major_vessels', 'thalassemia_type', 'target']
# cp - chest_pain_type
df.loc[df['chest_pain_type'] == 0, 'chest_pain_type'] = 'asymptomatic'
df.loc[df['chest_pain_type'] == 1, 'chest_pain_type'] = 'atypical angina'
df.loc[df['chest_pain_type'] == 2, 'chest_pain_type'] = 'non-anginal pain'
df.loc[df['chest_pain_type'] == 3, 'chest_pain_type'] = 'typical angina'
# restecg - rest_ecg_type
df.loc[df['rest_ecg_type'] == 0, 'rest_ecg_type'] = 'left ventricular hypertrophy'
df.loc[df['rest_ecg_type'] == 1, 'rest_ecg_type'] = 'normal'
df.loc[df['rest_ecg_type'] == 2, 'rest_ecg_type'] = 'ST-T wave abnormality'
# slope - st_slope_type
df.loc[df['st_slope_type'] == 0, 'st_slope_type'] = 'downsloping'
df.loc[df['st_slope_type'] == 1, 'st_slope_type'] = 'flat'
df.loc[df['st_slope_type'] == 2, 'st_slope_type'] = 'upsloping'
# thal - thalassemia_type
df.loc[df['thalassemia_type'] == 0, 'thalassemia_type'] = 'nothing'
df.loc[df['thalassemia_type'] == 1, 'thalassemia_type'] = 'fixed defect'
df.loc[df['thalassemia_type'] == 2, 'thalassemia_type'] = 'normal'
df.loc[df['thalassemia_type'] == 3, 'thalassemia_type'] = 'reversable defect'
data = pd.get_dummies(df, drop_first=False)
print(data.columns)
df_temp = data['thalassemia_type_fixed defect']
data = pd.get_dummies(df, drop_first=True)
print(data.head())
frames = [data, df_temp]
result = pd.concat(frames, axis=1)
result.drop('thalassemia_type_nothing', axis=1, inplace=True)
X = result.drop('target', axis=1)
y = result['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

logre = LogisticRegression()
logre.fit(X_train, y_train)
y_pred = logre.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
dt = DecisionTreeClassifier(random_state=42)

# fitting the model
dt.fit(X_train, y_train)

# calculating the predictions
y_pred = dt.predict(X_test)

# printing the test accuracy
print("The test accuracy score of Decision Tree is ",
      accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

# instantiating the object
rf = RandomForestClassifier()

# fitting the model
rf.fit(X_train, y_train)

# calculating the predictions
y_pred = dt.predict(X_test)

# printing the test accuracy
print("The test accuracy score of Random Forest is ",
      accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))

# instantiate the classifier
xgb_classifier = xgboost.XGBClassifier(n_estimators=50, random_state=42)

# fitting the model
xgb_classifier.fit(X_train, y_train)

# predicting values
y_pred = xgb_classifier.predict(X_test)
print("The test accuracy score of Gradient Boosting Classifier is ",
      accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'))
# instantiating the object and fitting
clf = SVC(kernel='linear', C=1, random_state=42).fit(X_train, y_train)

# predicting the values
y_pred = clf.predict(X_test)

# printing the test accuracy
print("The test accuracy score of SVM is ", accuracy_score(
    y_test, y_pred), f1_score(y_test, y_pred, average='macro'))
perm_importance = permutation_importance(
    clf, X_test, y_test, n_repeats=30, random_state=42)

# Get feature importances
rf_importances = rf.feature_importances_
xgb_importances = xgb_classifier.feature_importances_
svm_importances = perm_importance
logireg_importances = abs(logre.coef_[0])

# Create a DataFrame with feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'RandomForest': rf_importances,
    'XGBoost': xgb_importances,
    'logistice regression': logireg_importances,
    'svm': svm_importances.importances_mean,

})

display(feature_importances)
