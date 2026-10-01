# Task 10: Hyperparameter Tuning for an MLP
# Grid search / random search style comparison using sklearn MLPClassifier.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Baseline MLP
baseline = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_acc = accuracy_score(y_test, baseline_pred)
print(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}%")

# Grid Search
param_grid = {
    "hidden_layer_sizes": [(8,), (16,), (32,)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "learning_rate_init": [0.001, 0.01],
}

grid = GridSearchCV(
    MLPClassifier(max_iter=500, random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("\nBest Grid Search Parameters:")
print(grid.best_params_)

grid_pred = grid.best_estimator_.predict(X_test)
grid_acc = accuracy_score(y_test, grid_pred)
print(f"Grid Search Test Accuracy: {grid_acc * 100:.2f}%")

# Random Search
random_search = RandomizedSearchCV(
    MLPClassifier(max_iter=500, random_state=42),
    param_distributions=param_grid,
    n_iter=5,
    cv=3,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print("\nBest Random Search Parameters:")
print(random_search.best_params_)

random_pred = random_search.best_estimator_.predict(X_test)
random_acc = accuracy_score(y_test, random_pred)
print(f"Random Search Test Accuracy: {random_acc * 100:.2f}%")

print("\nComparison")
print(f"Baseline:    {baseline_acc * 100:.2f}%")
print(f"Grid Search: {grid_acc * 100:.2f}%")
print(f"Random Search: {random_acc * 100:.2f}%")
