# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data
data = pd.read_csv('Mall_Cust.csv')
val = data.iloc[:, [3, 4]].values
# y = data.iloc[:, 3].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(val, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the data
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(val)

# Visualising the clusters
plt.scatter(val[y_hc == 0, 0], val[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(val[y_hc == 1, 0], val[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(val[y_hc == 2, 0], val[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(val[y_hc == 3, 0], val[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(val[y_hc == 4, 0], val[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()