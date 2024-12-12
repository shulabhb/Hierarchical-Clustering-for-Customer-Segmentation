# Customer Segmentation Tutorial Using Hierarchical Clustering

## Project Overview
This project demonstrates the use of **Hierarchical Clustering** for customer segmentation in a retail setting. The goal is to cluster customers based on their **Age**, **Annual Income**, and **Spending Score** to derive actionable insights for targeted marketing and personalized customer experiences.

The dataset used for this project is the **Mall Customer Segmentation Data** (https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python), which includes the following features:
- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **Age**: Age of the customer.
- **Annual Income (k$)**: Annual income of the customer in thousands of dollars.
- **Spending Score (1-100)**: A score assigned based on customer spending behavior.

## Folder Structure
The project is organized as follows:

```
CustomerSegmentationTutorial/
|
|-- data/                    # Contains the raw and cleaned datasets
|   |-- Mall_Customers.csv   # Raw dataset
|   |-- cleaned_data.csv     # Cleaned and standardized dataset
|
|-- results/                 # Stores outputs such as visualizations and evaluation metrics
|   |-- visualizations/      # Contains plots and dendrograms
|
|-- scripts/                 # All Python scripts for different stages of the project
|   |-- download_data.py     # Script to download the dataset
|   |-- data_cleaning.py     # Cleans and standardizes the data
|   |-- eda.py               # Performs exploratory data analysis (EDA)
|   |-- hierarchical_clustering.py # Performs hierarchical clustering and saves dendrograms
|   |-- visualization.py        # Evaluates clusters and provides detailed visual analysis
|   |-- cluster_size.py      # Determines optimal cluster size using the elbow method
|
|-- README.md                # Documentation of the project
|-- requirements.txt         # Required Python libraries
|-- venv/                    # Python virtual environment
```

## Steps to Reproduce

### 1. Setup
#### Install Python Packages
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

#### Download Dataset
Run the following script to download the dataset from Kaggle (ensure your Kaggle API key is set up):
```bash
python scripts/download_data.py
```

### 2. Data Cleaning
Clean the data and standardize numeric features using the following script:
```bash
python scripts/data_cleaning.py
```
This will create a standardized dataset `cleaned_data.csv` in the `data/` folder.

### 3. Exploratory Data Analysis (EDA)
Perform EDA to visualize distributions, relationships, and insights:
```bash
python scripts/eda.py
```
The visualizations (histograms, box plots, scatter plots, and pie charts) will be saved in the `results/visualizations/` folder.

### 4. Clustering
Generate hierarchical clustering dendrograms:
```bash
python scripts/hierarchical_clustering.py
```
Dendrograms will be saved in the `results/visualizations/` folder.

### 5. Evaluate Clusters
Analyze the clusters and visualize detailed insights:
```bash
python scripts/evaluation.py
```
This script provides metrics like **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index** and prints detailed cluster-specific insights.

### 6. Optimal Cluster Size
Determine the optimal number of clusters using the elbow method:
```bash
python scripts/cluster_size.py
```
The elbow method plot will be saved in `results/visualizations/elbow_method.png`.

## Key Results
- **Hierarchical Clustering** grouped customers into distinct clusters based on their demographic and spending behavior.
- **Cluster Profiles** provide actionable insights for:
  - Targeted marketing strategies
  - Inventory management
  - Customer retention programs

## Real-Life Applications
1. **Targeted Marketing**: Segment customers for personalized promotions and campaigns.
2. **Customer Retention**: Identify at-risk customers and devise engagement strategies.
3. **Inventory Optimization**: Tailor inventory to meet the preferences of different customer groups.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`

## Contact
For queries or feedback, please reach out to the project developer:
- **Name**: Shulabh Bhattarai
- **Email**: shulabhb@gmail.com / sbhattarai_2026@depauw.edu / https://www.shulabhb.com/

## Future Work
1. Incorporate additional features like purchase history or product categories.
2. Extend clustering methods to include K-Means or DBSCAN for comparison.
3. Deploy the segmentation model in a web application to visualize customer clusters interactively.

# Hierarchical-Clustering-for-Customer-Segmentation
