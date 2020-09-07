import matplotlib.pyplot as plt
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

dataset_name = "Calls_67_X_B007_encoded_30e_(4x40x30)"
cluster_count = 2

pickle_in = open(dataset_name + ".pickle", "rb")
X = pickle.load(pickle_in)
X = X.reshape(len(X), 4800)

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=cluster_count, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Applying Multi Dimensional Scaling (MDS) to visualize results
embedding = MDS(n_components=2)
X_transformed = embedding.fit_transform(X)

# Visualising the clusters
# y_kmeans == cluster for the feature n
plt.scatter(X_transformed[y_kmeans == 0, 0], X_transformed[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_transformed[y_kmeans == 1, 0], X_transformed[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
if cluster_count > 2:
    plt.scatter(X_transformed[y_kmeans == 2, 0], X_transformed[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title(dataset_name + ' MDS for ' + str(cluster_count) + ' clusters')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.show()
