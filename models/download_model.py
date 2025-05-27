"""
Downloads the specified model from HuggingFace and saves it in the 
local cache.

Usage:  1. Install python dependencies
        2. Change the relevant ENV variables in env.py to your needs
        3. Run: python3 download_model.py

NOTE:   This should work with any model from HuggingFace available 
        through the transformers library (don't quote me on that).
"""

# Imports
from transformers import AutoTokenizer, AutoModel
import sys
import os

# Adds the parent directory to the system path so the env import works
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import get_env_var

# Load tokenizer and model from HuggingFace
model_name = get_env_var('MODEL_NAME')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save the tokenizer and model to the local cache
tokenizer.save_pretrained(os.path.join(os.path.dirname(__file__), model_name))
model.save_pretrained(os.path.join(os.path.dirname(__file__), model_name))

print(f"Model {model_name} downloaded and saved in the local cache.")