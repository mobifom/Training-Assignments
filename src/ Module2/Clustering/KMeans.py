import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

print("\n" + "=" * 60)
print("REAL-WORLD EXAMPLE: IRIS DATASET")
print("=" * 60)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"Iris Dataset: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
print(f"True classes: {iris.target_names}")

# Apply K-means with k=3 (we know there are 3 species)
kmeans_iris = KMeans(n_clusters=3, random_state=42)
labels_iris = kmeans_iris.fit_predict(X_iris)

# Evaluation metrics
ari_score = adjusted_rand_score(y_iris, labels_iris)
sil_score = silhouette_score(X_iris, labels_iris)

print(f"\nClustering Results:")
print(f"Silhouette Score: {sil_score:.3f}")
print(f"Adjusted Rand Index: {ari_score:.3f}")
print(f"Inertia: {kmeans_iris.inertia_:.2f}")

# Visualize using first two features
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=y_iris, cmap='viridis', alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('True Species Labels')

plt.subplot(1, 3, 2)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=labels_iris, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_iris.cluster_centers_[:, 0], 
           kmeans_iris.cluster_centers_[:, 1],
           marker='x', s=300, linewidths=3, color='red')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('K-means Clusters')

plt.subplot(1, 3, 3)
plt.scatter(X_iris[:, 2], X_iris[:, 3], c=labels_iris, cmap='viridis', alpha=0.7)
plt.scatter(kmeans_iris.cluster_centers_[:, 2], 
           kmeans_iris.cluster_centers_[:, 3],
           marker='x', s=300, linewidths=3, color='red')
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title('K-means Clusters (Petal Features)')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("K-MEANS TUTORIAL COMPLETE!")
print("=" * 60)

# Summary of key concepts
print("\nKey K-means Concepts:")
print("1. Centroid-based clustering algorithm")
print("2. Minimizes within-cluster sum of squares (WCSS/Inertia)")
print("3. Requires choosing k (number of clusters)")
print("4. Iterative algorithm: assign → update → repeat")
print("5. Sensitive to initialization and outliers")
print("6. Works best with spherical, well-separated clusters")

print("\nWhen to use K-means:")
print("✓ You have numeric data")
print("✓ Clusters are roughly spherical")
print("✓ You know approximate number of clusters")
print("✓ You need fast, scalable clustering")

print("\nWhen NOT to use K-means:")
print("✗ Clusters have very different sizes")
print("✗ Clusters have irregular shapes") 
print("✗ Data has many outliers")
print("✗ Features have very different scales")