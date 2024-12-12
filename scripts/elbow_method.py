"""
Elbow Method for K-Means Clustering

This script implements the Elbow Method to determine the optimal number of clusters for K-Means clustering in the Mall Customer Segmentation dataset.

Workflow:
1. Load the cleaned dataset.
2. Select numeric features for clustering (Age, Annual Income, Spending Score).
3. Calculate Within-Cluster Sum of Squares (WCSS) for different K values (number of clusters).
4. Implement the Elbow Method by plotting WCSS vs. K.
5. Analyze the plot to identify the "elbow" point, indicating the optimal K.
6. Save the Elbow Method plot for future reference.

"""

'####------------This file will give the plot for elbow method------------####'

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Defining data file path
CLEANED_DATA_PATH = "data/cleaned_data.csv"
RESULTS_DIR = "results/visualizations"

# Ensure results directory exists
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def calculate_wcss(data, max_clusters=10):
    """Calculate Within-Cluster Sum of Squares (WCSS) for different cluster counts."""
    wcss = []
    for n_clusters in range(1, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = clustering.fit_predict(data)

        # Calculating WCSS manually
        cluster_centers = [
            data[labels == cluster].mean(axis=0)
            for cluster in np.unique(labels)
        ]
        wcss_cluster = sum(
            np.sum((data[labels == cluster] - center) ** 2)
            for cluster, center in enumerate(cluster_centers)
        )
        wcss.append(wcss_cluster)
    return wcss

def plot_elbow_method(wcss):
    """Plot WCSS to visualize the elbow method."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(wcss) + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    elbow_path = os.path.join(RESULTS_DIR, "elbow_method.png")
    plt.savefig(elbow_path)
    plt.show()
    print(f"Elbow Method plot saved to: {elbow_path}")

if __name__ == "__main__":
    # Loading the cleaned dataset
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {CLEANED_DATA_PATH}")

    print("Loading cleaned data...")
    data = pd.read_csv(CLEANED_DATA_PATH)
    print("Cleaned data loaded successfully.")

    # Selecting numeric features for clustering
    numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = data[numeric_features].values

    print("Calculating WCSS for different cluster numbers...")
    # Calculating WCSS and plot the elbow method
    wcss = calculate_wcss(X, max_clusters=10)
    print("WCSS calculation complete. Plotting the Elbow Method...")
    plot_elbow_method(wcss)
    print("Elbow Method analysis completed!")