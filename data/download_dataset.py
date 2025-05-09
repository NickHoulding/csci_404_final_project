"""
Downloads the dataset from Hugging Face and saves it as a CSV file.

Usage:  1. Install dependencies
        2. Run: cd data
        3. Change the global variables to your needs
        4. Run: python3 download_dataset.py

NOTE:   Your current working directory must be the same as 
        this script's directory.
"""

# Imports
from datasets import load_dataset

# Globals
DATASET_NAME = "qiaojin/PubMedQA"
FILE_NAME = "PubMedQA.csv"

dataset = load_dataset(DATASET_NAME, "pqa_artificial")
dataset["train"].to_csv(FILE_NAME, index=False)