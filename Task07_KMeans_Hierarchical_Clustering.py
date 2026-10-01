# Task 7: Partitioning and Hierarchical Clustering
# Iris dataset, as used in the laboratory manual.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

scaler = StandardScaler()
X = scaler.fit_transform(df)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage="ward")
hierarchical_labels = hierarchical.fit_predict(X)

print("K-Means Clustering")
print("Silhouette Score:", round(silhouette_score(X, kmeans_labels), 4))
print("Davies-Bouldin Index:", round(davies_bouldin_score(X, kmeans_labels), 4))

print("\nHierarchical Clustering")
print("Silhouette Score:", round(silhouette_score(X, hierarchical_labels), 4))
print("Davies-Bouldin Index:", round(davies_bouldin_score(X, hierarchical_labels), 4))

# Dendrogram
plt.figure(figsize=(10, 5))
Z = linkage(X, method="ward")
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# Scatter comparison
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels)
plt.title("K-Means Clustering")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=hierarchical_labels)
plt.title("Hierarchical Clustering")

plt.tight_layout()
plt.show()
