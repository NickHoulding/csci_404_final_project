"""
Models module for MediChat: Handling model loading and inference
"""

# Imports
from env import get_env_var
import ollama

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
    )
    
    return response.message.content