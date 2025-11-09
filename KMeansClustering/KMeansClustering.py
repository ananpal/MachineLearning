import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, data):
        self.dataSet = [data]
        self.centroid = np.array(data)
    
    def add_data(self, data):
        self.dataSet.append(data)
        self.centroid = np.mean(self.dataSet, axis=0)
    
    def clear(self):
        self.dataSet = []

class KMeansCluster:
    def __init__(self, k, max_iters=100):
        self.k = k
        self.k_clusters = []
        self.max_iters = max_iters
    
    def _initialize(self, X):
        # Randomly select k data points as initial centroids
        n_samples = X.shape[0]
        k_random_indices = np.random.choice(n_samples, size=self.k, replace=False)
        self.k_clusters = []
        for idx in k_random_indices:
            self.k_clusters.append(Cluster(X[idx]))
        return k_random_indices
        
    def _assign_cluster(self, X):
        # Store old centroids for distance calculations
        old_centroids = [cluster.centroid.copy() for cluster in self.k_clusters]
        
        # Clear previous assignments
        for cluster in self.k_clusters:
            cluster.clear()
        
        # Assign each point to nearest cluster (using old centroids)
        for x in X:
            distances = [np.linalg.norm(x - old_centroid) for old_centroid in old_centroids]
            nearest_cluster = np.argmin(distances)
            self.k_clusters[nearest_cluster].add_data(x)
   
    def _preprocessing(self, x, is_train = False):
        if is_train:
            self.mean = np.mean(x, axis=0)  # Mean per feature
            self.std = np.std(x, axis=0)     # Std per feature (not sd)
            # Avoid division by zero
            self.std[self.std == 0] = 1
        return (x - self.mean) / self.std

    def fit(self, x):
        x = self._preprocessing(x, is_train=True)
        self._initialize(x)
        
        # Iterate until convergence or max iterations
        for iteration in range(self.max_iters):
            old_centroids = np.array([cluster.centroid.copy() for cluster in self.k_clusters])
            self._assign_cluster(x)
            new_centroids = np.array([cluster.centroid for cluster in self.k_clusters])
            
            # Check for convergence (centroids haven't changed)
            if np.allclose(old_centroids, new_centroids, atol=1e-4):
                break
    

    def predict(self, X):
        X = self._preprocessing(X)
        labels = []
        X = np.array(X)
        for x in X:
            distances = [np.linalg.norm(x - cluster.centroid) for cluster in self.k_clusters]
            labels.append(np.argmin(distances))
        return np.array(labels)
    
    def calculate_inertia(self, X):
        """Calculate within-cluster sum of squares (inertia/WCSS)"""
        X_preprocessed = self._preprocessing(X, is_train=False)
        # Get labels directly from preprocessed data
        labels = []
        for x in X_preprocessed:
            distances = [np.linalg.norm(x - cluster.centroid) for cluster in self.k_clusters]
            labels.append(np.argmin(distances))
        labels = np.array(labels)
        
        inertia = 0
        for i, cluster in enumerate(self.k_clusters):
            cluster_points = X_preprocessed[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - cluster.centroid) ** 2)
        return inertia
    
    def plot_clusters(self, X, labels=None, title="K-Means Clustering", save_path="KMeansClustering/cluster_visualization.png"):
        """Visualize clusters with data points and centroids"""
        X_original = np.array(X)  # Original scale data
        if labels is None:
            labels = self.predict(X_original)  # predict handles preprocessing internally
        
        # Convert centroids back to original scale for visualization
        centroids_original = np.array([cluster.centroid * self.std + self.mean for cluster in self.k_clusters])
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, self.k))
        
        # Plot data points colored by cluster
        for i in range(self.k):
            cluster_points = X_original[labels == i]
            if len(cluster_points) > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=50)
        
        # Plot centroids
        plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_elbow_method(self, X, k_range=range(2, 11), save_path="KMeansClustering/elbow_method.png"):
        """Plot elbow method to determine optimal k"""
        inertias = []
        for k in k_range:
            temp_model = KMeansCluster(k, max_iters=100)
            temp_model.fit(X)
            inertias.append(temp_model.calculate_inertia(X))
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (WCSS)')
        plt.title('Elbow Method For Optimal k')
        plt.grid(True, alpha=0.3)
        plt.xticks(k_range)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return inertias

data_pd = pd.read_csv("KMeansClustering/Mall_Customers.csv")

# Select only numeric columns (exclude any string/categorical columns)
data = data_pd[['Annual Income (k$)', 'Spending Score (1-100)']]
data = data.to_numpy()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

k=5
model = KMeansCluster(k)
model.fit(train_data)

# Calculate and print inertia
inertia = model.calculate_inertia(train_data)
print(f"\nInertia (WCSS) for k={k}: {inertia:.2f}")

# Visualize clusters
print("\nGenerating cluster visualization...")
model.plot_clusters(train_data, title=f"K-Means Clustering (k={k})")

# Optional: Run elbow method to find optimal k
print("\nRunning elbow method analysis...")
inertias = model.plot_elbow_method(train_data, k_range=range(2, 11))

