"""
Downloads the dataset from Hugging Face and saves it as a CSV file.

NOTE:   Your current working directory must be the same as 
        this script's directory.
"""

from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
dataset["train"].to_csv("PubMedQA.csv")