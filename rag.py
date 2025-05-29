"""
Takes the user query and retrieves the top k most relevant texts 
from the knowledge base.
"""

# Imports
import types
import sys

# Add torch.classes to sys.modules to avoid streamlit errors
if 'torch.classes' not in sys.modules:
    sys.modules['torch.classes'] = types.ModuleType('torch.classes')

from transformers import AutoTokenizer, AutoModel
from knowledge_base import knowledgeBase
from env import get_env_var
import numpy as np
import pickle
import torch
import os

# Globals
# Define consistent context prompt format
PROMPT_TEMPLATE = """
Based only on the following medical context, provide a concise 
clinical insight to help a doctor interpret a patient's symptoms. 
The insight must be medically relevant, grounded in the provided 
context, and limited to a single paragraph or short list.

{context}

---
Patient presents with the following symptoms:
{prompt}

Respond in a clear, medically appropriate tone. Do not speculate 
or provide a diagnosis.
"""

model_path = os.path.join(
    os.path.dirname(__file__), 
    'models', 
    get_env_var("EMBEDDING_MODEL")
)
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "cpu"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Disable training-specific features
model.eval()

# Take advantage of GPU if available
if torch.cuda.is_available():
    model.to(device)

def load_kb(file_path: str) -> knowledgeBase:
    """
    Load a knowledge base from a pickle file.

    Args:
        file_path (str): The path to the pickle file 
            containing the knowledge base.
    Returns:
        knowledgeBase: An instance of the knowledgeBase 
            class containing the loaded data.
    """
    try:
        with open(file_path, 'rb') as f:
            kb = pickle.load(f)
        return kb

    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    
    except NotADirectoryError:
        print(f"Path error: {file_path} contains an invalid directory.")
        return None
    
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return None

def get_context_prompt(
        user_query: str, 
        results: list[dict]
) -> str:
    """
    Create a formatted context prompt for the model.

    Args:
        user_query (str): The user's query.
        results (list[dict]): The top k results from the 
            knowledge base.
    Returns:
        str: The formatted context prompt.
    """
    context = "\n".join([
        f"**{result['text']}**\n" 
        for result in results
    ])

    return PROMPT_TEMPLATE.format(
        context=context, 
        prompt=user_query
    )

def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding of a given text.

    Args:
        text (str): The text to get the embedding for.
    Returns:
        np.ndarray: The embedding for the text.
    """
    # Tokenize the text
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Get the inputs
    inputs = {
        name: tensor.to(device) 
        for name, tensor in inputs.items()
    }

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Normalize and return the embeddings
    embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embed = embed / np.linalg.norm(
        embed, 
        axis=1, 
        keepdims=True
    )

    return embed