"""
Takes the user query and retrieves the top k most relevant texts 
from the FAISS knowledge base.
"""

# Imports
import types
import sys

# Add torch.classes to sys.modules to avoid streamlit errors
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
model_path = os.path.join(
    os.path.dirname(__file__), 
    'models', 
    get_env_var("EMBEDDING_MODEL")
)
knowledge_save_path = os.path.join(
    os.path.dirname(__file__), 
    'knowledge'
)
faiss_index = faiss.read_index(
    os.path.join(
        knowledge_save_path, 
        'index.faiss'
    )
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

# Load the texts from the knowledge base
text_path = os.path.join(knowledge_save_path, "texts.pkl")
with open(text_path, "rb") as f:
    texts = pickle.load(f)

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

def search_kb(
        embedding: np.ndarray, 
        top_k: int = 3
) -> list[dict]:
    """
    Search the knowledge base for the most relevant texts.

    Args:
        embedding (np.ndarray): The embedding of the query text.
        top_k (int): The number of top results to return.
    Returns:
        list[dict]: A list of dictionaries containing the text 
            and score for each result.
    """

    """
    Performs nearest neighbor search on the FAISS index using 
    normalized embeddings. Since FAISS uses Euclidean distance 
    by default, the vector normalization in get_embedding() 
    ensures mathematical equivalence to cosine similarity 
    search, as Euclidean distance between normalized vectors 
    directly correlates with their cosine similarity.
    """
    scores, indices = faiss_index.search(embedding, top_k)

    # Get the top k relevant chunks and their scores
    results = []
    for i in range(top_k):
        ctx_idx = indices[0][i]
        score = scores[0][i]

        results.append({
            "text": texts[ctx_idx],
            "score": score
        })

    return results