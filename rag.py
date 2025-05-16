"""
Takes the user query and retrieves the top k most relevant texts 
from the FAISS knowledge base.
"""

# Imports
import types
import sys

# Add torch.classes to sys.modules to avoid streamlit errors:
if 'torch.classes' not in sys.modules:
    sys.modules['torch.classes'] = types.ModuleType('torch.classes')

from transformers import AutoTokenizer, AutoModel
from env import get_env_var
import numpy as np
import pickle
import faiss
import torch
import os

# Globals
embedding_model_path = os.path.join(os.path.dirname(__file__), 
                                    'models', 
                                    get_env_var("EMBEDDING_MODEL"))
knowledge_save_path = os.path.join(os.path.dirname(__file__), 
                                   'knowledge')
faiss_index = faiss.read_index(os.path.join(knowledge_save_path, 
                                            'index.faiss'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
model = AutoModel.from_pretrained(embedding_model_path)
model.to(device)
model.eval()

with open(os.path.join(knowledge_save_path, "texts.pkl"), "rb") as f:
    texts = pickle.load(f)

def search_top_k(query_text: str, k=3) -> list[dict]:
    """
    Search for the top k most relevant texts for a given query text.

    Args:
        query_text (str): The query text to search on.
        k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the text and 
            score for each result.
    """
    # Tokenize the query text
    inputs = tokenizer(query_text, 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True)
    
    # Get the inputs
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Normalize the embeddings
    q_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    q_embed = q_embed / np.linalg.norm(q_embed, axis=1, keepdims=True)
    scores, indices = faiss_index.search(q_embed, k)

    # Get the top k results
    results = []
    for i in range(k):
        ctx_idx = indices[0][i]
        score = scores[0][i]
        results.append({
            "text": texts[ctx_idx],
            "score": score
        })

    return results