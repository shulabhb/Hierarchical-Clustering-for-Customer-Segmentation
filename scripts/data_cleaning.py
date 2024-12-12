 

"""
Data Cleaning Script

Purpose:
This script handles the data cleaning and preprocessing of the Mall Customer Segmentation dataset. 
It includes loading the dataset, performing basic data exploration, standardizing key numeric features(age, annual income and spending score), 
and saving the cleaned data for further analysis.

Workflow:
1. Load the dataset from the data folder.
2. Explore the dataset by checking its structure, summary statistics, and missing values.
3. Standardize numerical features such as Age, Annual Income, and Spending Score.
4. Save the cleaned dataset for future use in clustering and analysis.
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Defining file paths
DATA_PATH = "data/Mall_Customers.csv"  # Original dataset path
CLEANED_DATA_PATH = "data/cleaned_data.csv"  # Path to save the cleaned data

def load_data(file_path):
    """
    Load the dataset into a DataFrame.
    Checks if the file exists and loads it using Pandas.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    print("Loading dataset...")
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    return data

def explore_data(data):
    """
    Perform basic data exploration:
    - Display dataset information (columns, data types)
    - Print summary statistics (mean, min, max, etc.)
    - Check for missing values
    """
    print("\n--- Basic Data Overview ---")
    print(data.info())

    print("\n--- Summary Statistics ---")
    print(data.describe())

    print("\n--- Checking for Missing Values ---")
    print(data.isnull().sum())

def clean_and_standardize(data):
    """
    Standardize numeric features for clustering:
    - Features to standardize: Age, Annual Income, Spending Score
    - Use StandardScaler from scikit-learn for scaling.
    - Return a new DataFrame with standardized numeric features.
    """
    numeric_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    # Initializing StandardScaler
    scaler = StandardScaler()

    # Standardizing numeric features
    data_scaled = data.copy()
    data_scaled[numeric_features] = scaler.fit_transform(data[numeric_features])
    
    print("Data standardization complete.")
    return data_scaled

def save_cleaned_data(data, output_path):
    """
    Save the cleaned dataset to a specified CSV file.
    The cleaned file will be used for clustering and analysis.
    """
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Loading and exploring the dataset
    data = load_data(DATA_PATH)
    explore_data(data)
    
    # Standardizing numeric features for better clustering
    cleaned_data = clean_and_standardize(data)
    
    # Saving the cleaned dataset for future use
    save_cleaned_data(cleaned_data, CLEANED_DATA_PATH)
