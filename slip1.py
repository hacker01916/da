##Q1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    'Product': ['Laptop', 'Mobile', 'Tablet', 'TV', 'Camera', 'Watch'],
    'Price': [50000, 20000, 15000, 40000, 30000, 5000],
    'Rating': [4.5, 4.2, 4.0, 4.4, 4.3, 3.8],
    'Sales': [100, 250, 180, 120, 90, 300]
}

df = pd.DataFrame(data)      
print("Product DataFrame:")
print(df)

X = df[['Price', 'Rating', 'Sales']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z, labels=df['Product'].values)
plt.title("Product Clustering Dendrogram")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()



#Q2

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[:, :2]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.scatter(
    X[:, 0], X[:, 1],
    c=labels, cmap='viridis',
    s=50, label='Data Points'
)
 
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c='red', marker='X', s=200,
    label='Centroids'
)

plt.title("K-Means Clustering (K=3)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()

print("Cluster Centroids:")
print(centroids)
