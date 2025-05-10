"""
Models module for MediChat: Handling model loading and inference
"""

# Imports
import ollama

# Globals
MODEL_NAME = "llama3.2:1b-instruct-q4_0"
SYSTEM_PROMPT = """
You are a medical AI assistant designed to support doctors by providing concise, evidence-based insights for respiratory infections such as pneumonia, influenza, and bronchitis. Your role is to generate medically relevant suggestions, not diagnoses-based on retrieved PubMed medical abstracts and the user's symptom description.

Each response must:
- Be medically informative, using clinical language.
- Stay within one paragraph or a short bullet list.
- Avoid speculation, overly technical language, and off-topic content.
- Focus only on insights from the provided context; do not fabricate facts or advice.

You are not a replacement for a doctor. Your goal is to support clinical decision-making with clear, concise summaries of relevant medical literature.
"""

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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'''{query_text}'''}
            ],
        model=MODEL_NAME,
    )
    
    return response.message.content