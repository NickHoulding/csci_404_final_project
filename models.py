"""
Handles model loading and inference.
"""

# Imports
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel
from env import get_env_var
import numpy as np
import ollama
import types
import sys
import os

# Add torch.classes to sys.modules to avoid streamlit errors
if 'torch.classes' not in sys.modules:
    sys.modules['torch.classes'] = types.ModuleType('torch.classes')

import torch

# Globals
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

class ModelResponseStructure(BaseModel):
    """
    Model response structure.
    """
    topic_title: str
    response: str

def query_model(query_text: str) -> str:
    """
    Query the model with a given query string.

    Args:
        query_text (str): The user's query.
    Returns:
        response (str): The model's response to the query.
    """
    response = ollama.chat(
        messages=[
            {"role": "system", "content": get_env_var('SYSTEM_PROMPT')},
            {"role": "user", "content": f'''{query_text}'''}
        ],
        model=get_env_var('GEN_MODEL_NAME'),
        format=ModelResponseStructure.model_json_schema()
    )

    response_obj = ModelResponseStructure.model_validate_json(
        response.message.content
    )
    
    return (response_obj.topic_title,
            response_obj.response)

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