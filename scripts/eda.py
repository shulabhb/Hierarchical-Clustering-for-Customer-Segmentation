"""
Exploratory Data Analysis (EDA) Script

This script performs exploratory data analysis on the Mall Customer Segmentation dataset.
It visualizes the distribution of key features (Age, Annual Income, Spending Score) and relationships between them.
It also visualizes the gender distribution in the dataset.

Workflow:
1. Loads the cleaned dataset.
2. Visualizes feature distributions using histograms and box plots.
3. Creates scatter plots to visualize relationships between features.
4. Visualizes gender distribution using a pie chart.
5. Saves the generated visualizations to the 'results/visualizations' directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Defined file paths
CLEANED_DATA_PATH = "data/Mall_Customers.csv"
EDA_OUTPUT_DIR = "results/visualizations"

# Ensuring output directory exists
if not os.path.exists(EDA_OUTPUT_DIR):
    os.makedirs(EDA_OUTPUT_DIR)
# Loading the original data file
def load_original_data(file_path):
    """Load the cleaned dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cleaned data not found at {file_path}")
    print("Loading cleaned dataset...")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return data

# Visualization Functions
def plot_distribution(data, feature, output_dir):
    """Visualize feature distribution using histograms and box plots."""
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(data[feature], kde=True, color="skyblue")
    plt.title(f'{feature} Distribution - Histogram')
    
    # Box Plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data[feature], color="lightgreen")
    plt.title(f'{feature} Distribution - Box Plot')
    
    output_file = os.path.join(output_dir, f'{feature}_distribution.png')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_scatter(data, x_feature, y_feature, output_dir):
    """Create scatter plots between features."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[x_feature], y=data[y_feature], hue=data['Gender'], palette="coolwarm")
    plt.title(f'{x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)

    output_file = os.path.join(output_dir, f'{x_feature}_vs_{y_feature}.png')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")


def plot_gender_distribution(data, output_dir):
    """Visualize gender distribution using a pie chart."""
    gender_counts = data['Gender'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightpink'])
    plt.title("Gender Distribution")

    output_file = os.path.join(output_dir, 'gender_distribution.png')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    # Loading and visualizing the data
    data = load_original_data(CLEANED_DATA_PATH)

    # Feature Distributions
    plot_distribution(data, 'Age', EDA_OUTPUT_DIR)
    plot_distribution(data, 'Annual Income (k$)', EDA_OUTPUT_DIR)
    plot_distribution(data, 'Spending Score (1-100)', EDA_OUTPUT_DIR)

    # Scatter Plots
    plot_scatter(data, 'Age', 'Spending Score (1-100)', EDA_OUTPUT_DIR)
    plot_scatter(data, 'Annual Income (k$)', 'Spending Score (1-100)', EDA_OUTPUT_DIR)
    plot_scatter(data, 'Age', 'Annual Income (k$)', EDA_OUTPUT_DIR)

    # Gender Distribution
    plot_gender_distribution(data, EDA_OUTPUT_DIR)

    print("EDA visualizations completed.")
