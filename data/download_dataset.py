"""
Downloads the dataset from Hugging Face and saves it as a CSV file.

Usage:  1. Install python dependencies
        2. Change the relevant ENV variables in env.py to your needs
        3. Run: python3 download_dataset.py

NOTE:   Your current working directory must be the same as 
        this script's directory.
"""

# Imports
from datasets import load_dataset
import sys
import os

# Adds the parent directory to the system path so the env import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import get_env_var

# Download and save the dataset
dataset = load_dataset(get_env_var('DATASET_NAME'), get_env_var('DATASET_SUBSET'))
dataset["train"].to_csv(get_env_var('FILE_NAME'), index=False)