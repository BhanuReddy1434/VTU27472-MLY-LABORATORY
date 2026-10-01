# Task 3: Principal Component Analysis (PCA) vs Linear Discriminant Analysis (LDA)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_classification(
    n_samples=500,
    n_features=50,
    n_informative=10,
    n_redundant=10,
    n_classes=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for c in np.unique(y_train):
    axes[0].scatter(
        X_train_pca[y_train == c, 0],
        X_train_pca[y_train == c, 1],
        label=f"Class {c}"
    )
axes[0].set_title("PCA Visualization")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend()

for c in np.unique(y_train):
    axes[1].scatter(
        X_train_lda[y_train == c, 0],
        X_train_lda[y_train == c, 1],
        label=f"Class {c}"
    )
axes[1].set_title("LDA Visualization")
axes[1].set_xlabel("Linear Discriminant 1")
axes[1].set_ylabel("Linear Discriminant 2")
axes[1].legend()

plt.tight_layout()
plt.show()

def evaluate(Xtr, Xte, name):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    score = accuracy_score(y_test, pred)
    print(f"{name} Accuracy: {score:.2f}")

evaluate(X_train, X_test, "Original")
evaluate(X_train_pca, X_test_pca, "PCA")
evaluate(X_train_lda, X_test_lda, "LDA")
