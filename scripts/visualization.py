"""
Evaluation Script

This script analyzes and visualizes the clusters obtained from the Mall Customer Segmentation dataset.

Workflow:
1. Loads the clustered dataset.
2. Analyzes the clusters by calculating cluster size, average age, annual income, and spending score.
3. Visualizes the clusters using scatter plots:
   - Age vs. Spending Score
   - Annual Income vs. Spending Score
4. Saves the cluster analysis and visualizations to the 'results/evaluation' directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
 
CLUSTER_OUTPUT_DIR = "results/clustering"
EVALUATION_OUTPUT_DIR = "results/evaluation"
CLUSTERED_DATA_PATH = os.path.join(CLUSTER_OUTPUT_DIR, "customer_segmentation_clusters.csv")

# Ensure output directory exists
if not os.path.exists(EVALUATION_OUTPUT_DIR):
    os.makedirs(EVALUATION_OUTPUT_DIR)

def load_clustered_data(file_path):
    """Loading the dataset with cluster labels."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Clustered data not found at {file_path}")
    print("Loading clustered data...")
    data = pd.read_csv(file_path)
    print("Clustered data loaded successfully.")
    return data

def analyze_clusters(data):
    """Analyzing clusters and saving results."""
    print("\nAnalyzing Clusters...")
    cluster_summary = data.groupby('Cluster').agg(
        Cluster_Size=('CustomerID', 'count'),
        Avg_Age=('Age', 'mean'),
        Avg_Annual_Income=('Annual Income (k$)', 'mean'),
        Avg_Spending_Score=('Spending Score (1-100)', 'mean')
    ).reset_index()

    print("Cluster Summary:\n", cluster_summary)

    # Saving cluster analysis to a CSV file
    summary_path = os.path.join(EVALUATION_OUTPUT_DIR, "cluster_analysis.csv")
    cluster_summary.to_csv(summary_path, index=False)
    print(f"Cluster analysis saved to: {summary_path}")

    return cluster_summary

def visualize_clusters(data):
    """Visualizing clusters in 2D scatter plots."""
    print("\nVisualizing clusters...")

    # Scatter plot for Age vs Spending Score 
    # (The plots show the actual age and the printed stats show the standarized metrics used to avoid big calculations and optimal graph plots.)
    plt.figure(figsize=(10, 6))
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(
            cluster_data['Age'], cluster_data['Spending Score (1-100)'],
            label=f'Cluster {cluster}', alpha=0.7
        )
    plt.title('Clusters: Age vs Spending Score')
    plt.xlabel('Age')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    scatter_path = os.path.join(EVALUATION_OUTPUT_DIR, "age_vs_spending_clusters.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved to: {scatter_path}")

    # Scatter plot for Annual Income vs Spending Score
    plt.figure(figsize=(10, 6))
    for cluster in data['Cluster'].unique():
        cluster_data = data[data['Cluster'] == cluster]
        plt.scatter(
            cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
            label=f'Cluster {cluster}', alpha=0.7
        )
    plt.title('Clusters: Annual Income vs Spending Score')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True)
    scatter_path = os.path.join(EVALUATION_OUTPUT_DIR, "income_vs_spending_clusters.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"Scatter plot saved to: {scatter_path}")

if __name__ == "__main__":
    # Loading clustered data
    data = load_clustered_data(CLUSTERED_DATA_PATH)

    # Analyzing clusters
    cluster_summary = analyze_clusters(data)

    # Visualizing clusters
    visualize_clusters(data)

    print("Evaluation completed! Check the evaluation folder for results.")
