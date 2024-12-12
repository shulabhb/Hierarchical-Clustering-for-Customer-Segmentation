"""
Download Script for Mall Customer Segmentation Dataset

This script is intended to download the original Mall Customer Segmentation dataset from Kaggle or do it manually here: 
https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python .

**Important Note:**

* Only run this script if you don't have the dataset already.
* Make sure you have initialized Kaggle and set up your Kaggle API before running this script.

Workflow:
1. Define the Kaggle dataset path and local download directory.
2. Create the data directory if it doesn't exist.
3. Download the dataset using the Kaggle API (subprocess module).
4. Print confirmation messages for download completion.
"""
import os
import subprocess

def download_dataset():
    # Defining the Kaggle dataset path and local download directory
    kaggle_dataset = "vjchoudhary7/customer-segmentation-tutorial-in-python"
    data_dir = os.path.join(os.getcwd(), "data")

    # Ensuring the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")

    # Downloading the dataset using Kaggle API
    print("Downloading dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_dataset, "--unzip", "-p", data_dir],
        check=True
    )
    print(f"Dataset downloaded and saved to {data_dir}")

if __name__ == "__main__":
    download_dataset()
