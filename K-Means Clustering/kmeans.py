# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Cust.csv')
val = dataset.iloc[:, [3,4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wc = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(val)
    wc.append(kmeans.inertia_)
plt.plot(range(1, 11), wc)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_pred = kmeans.fit_predict(val)

# Visualising the clusters
plt.scatter(val[y_pred == 0, 0], val[y_pred == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(val[y_pred == 1, 0], val[y_pred == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(val[y_pred == 2, 0], val[y_pred == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(val[y_pred == 3, 0], val[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(val[y_pred == 4, 0], val[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()