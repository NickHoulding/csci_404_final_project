"""
Downloads the dataset from Hugging Face and saves it as a CSV file.

Usage:  1. Install dependencies
        2. Run: cd data
        3. Run: python3 download.py

NOTE:   Your current working directory must be the same as 
        this script's directory.
"""

# Imports
from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
dataset["train"].to_csv("PubMedQA.csv")