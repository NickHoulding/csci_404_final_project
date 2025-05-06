"""
Models module for MediChat: handling model loading and inference
"""

# Imports
import ollama

# Globals
MODEL_NAME = "llama3.2:1b-instruct-q4_0"

def query_model(query_text: str) -> str:
    """
    Query the model with a given query string.

    Args:
        query_text (str): The user's query.
    Returns:
        response (str): The model's response to the query.
    """
    # Query the model
    response = ollama.chat(
        messages=[{
                "role": "user",
                "content": f'''
                    {query_text}
                '''}],
        model=MODEL_NAME,
    )
    
    # Return the response
    return response.message.content