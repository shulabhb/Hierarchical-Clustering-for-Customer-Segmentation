import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Defining file Paths
CLEANED_DATA_PATH = "data/cleaned_data.csv"
CLUSTER_OUTPUT_DIR = "results/clustering"

# Ensuring output directory exists
if not os.path.exists(CLUSTER_OUTPUT_DIR):
    os.makedirs(CLUSTER_OUTPUT_DIR)

def load_cleaned_data(file_path):
    """Loading the cleaned dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned data not found at {file_path}")
    print("Loading cleaned dataset...")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return data

def cluster_data(X, labels, output_dir, plot_title, file_prefix, n_clusters=10, linkage_method='ward'):
    """Performing hierarchical clustering and save dendrogram."""
    print(f"\nPerforming clustering for {plot_title}...")
    Z = linkage(X, method=linkage_method)

    # Plot Dendrogram
    plt.figure(figsize=(16, 10))
    plt.title(plot_title)
    plt.subplots_adjust(bottom=0.3)
    dendrogram(
        Z, labels=labels, leaf_rotation=90, leaf_font_size=10, truncate_mode=None,
        show_contracted=False, color_threshold=1.5 * max(Z[:, 2])
    )
    plt.ylabel("Distance")
    plt.xlabel("Customer IDs")
    dendrogram_path = os.path.join(output_dir, f'{file_prefix}_detailed_dendrogram.png')
    plt.tight_layout()
    plt.savefig(dendrogram_path)
    plt.close()
    print(f"Detailed Dendrogram saved to: {dendrogram_path}")

    # Performing Agglomerative Clustering
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    cluster_labels = model.fit_predict(X)
    return cluster_labels

def save_cluster_results(data, cluster_labels, output_dir, file_prefix):
    """Save clustering results to CSV."""
    data['Cluster'] = cluster_labels
    output_file = os.path.join(output_dir, f'{file_prefix}_clusters.csv')
    data.to_csv(output_file, index=False)
    print(f"Cluster results saved to: {output_file}\n")

if __name__ == "__main__":
    # Loading cleaned data
    data = load_cleaned_data(CLEANED_DATA_PATH)

    # Performing clustering on numeric features
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = data[features].values

    # Performing Clustering
    cluster_labels = cluster_data(
        X, labels=data['CustomerID'].astype(str).values, output_dir=CLUSTER_OUTPUT_DIR,
        plot_title="Hierarchical Clustering Dendrogram", file_prefix="customer_segmentation",
        n_clusters=4, linkage_method='ward'
    )

    # Saveing Clustering Results
    save_cluster_results(data, cluster_labels, CLUSTER_OUTPUT_DIR, "customer_segmentation")

    print("Clustering process completed! Proceed to evaluation.py for detailed analysis.")
