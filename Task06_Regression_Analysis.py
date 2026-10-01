# Task 6: Regression Analysis
# Includes simple linear regression, multiple regression/EDA,
# multicollinearity check, and logistic regression as described in the manual.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Simple linear regression using a synthetic predictor
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([2, 4, 5, 8, 10, 12])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
pred = linear_model.predict(X_test)

print("Simple Linear Regression")
print("R-squared:", r2_score(y_test, pred))
print("MSE:", mean_squared_error(y_test, pred))
print("Prediction for x=7:", linear_model.predict([[7]])[0])

# Multiple regression + correlation / VIF on Breast Cancer data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

X_multi = df.drop(columns="target")
y_multi = df["target"]

print("\nCorrelation matrix:")
print(X_multi.corr())

# VIF for first 10 features to keep the demonstration manageable
X_vif = sm.add_constant(X_multi.iloc[:, :10])
vif = pd.DataFrame({
    "Feature": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ]
})
print("\nVIF:")
print(vif)

# Logistic regression
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
