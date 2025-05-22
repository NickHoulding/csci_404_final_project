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
embedding_model_path = os.path.join(os.path.dirname(__file__), 'models', get_env_var("EMBEDDING_MODEL"))
knowledge_save_path = os.path.join(os.path.dirname(__file__), 'knowledge')
faiss_index = faiss.read_index(os.path.join(knowledge_save_path, 'index.faiss'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
model = AutoModel.from_pretrained(embedding_model_path)
model.to(device)
model.eval()

with open(os.path.join(knowledge_save_path, "texts.pkl"), "rb") as f:
    texts = pickle.load(f)

PROMPT_TEMPLATE = """
Based only on the following medical context, provide a concise clinical insight to help a doctor interpret a patient's symptoms. The insight must be medically relevant, grounded in the provided context, and limited to a single paragraph or short list.

{context}

---
Patient presents with the following symptoms:
{prompt}

Respond in a clear, medically appropriate tone. Do not speculate or provide a diagnosis.
"""

def get_context_prompt(user_query: str) -> str:
    """
    Create a formatted context prompt for the model.

    Args:
        prompt (str): The user query.

    Returns:
        str: The formatted prompt.
    """
    results = searck_kb(user_query, top_k=3)
    context = "\n".join([f"**{result['text']}**\n" for result in results])

    return PROMPT_TEMPLATE.format(
        context=context, 
        prompt=user_query
    )

def get_embedding(text: str) -> np.ndarray:
    """
    Get the embedding for a given text.

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
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Normalize the embeddings
    embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)

    return embed

def searck_kb(query_text: str, top_k=3) -> list[dict]:
    """
    Search the knowledge base for the most relevant texts.

    Args:
        query_text (str): The query text to search on.
        top_k (int): The number of top results to return.

    Returns:
        list[dict]: A list of dictionaries containing the text and 
            score for each result.
    """
    # Get the embedding for the query text
    q_embed = get_embedding(query_text)
    scores, indices = faiss_index.search(q_embed, top_k)

    # Get the top k results
    results = []
    for i in range(top_k):
        ctx_idx = indices[0][i]
        score = scores[0][i]
        results.append({
            "text": texts[ctx_idx],
            "score": score
        })

    return results