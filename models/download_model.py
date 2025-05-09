"""
Downloads the specified model from HuggingFace and 
saves it in the local cache.

Usage:  1. Install dependencies
        2. Run: cd models
        3. Change the global variable to your needs
        4. Run: python3 download_model.py

NOTE:   Some HuggingFace models have different imports from 
        transformers. Check the model's documentation and 
        add their imports below.
"""

# Imports
from transformers import AutoTokenizer, AutoModel

# Globals
MODEL_NAME = "dmis-lab/biobert-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(MODEL_NAME)
model.save_pretrained(MODEL_NAME)