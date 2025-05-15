"""
Models module for MediChat: Handling model loading and inference
"""

# Imports
from pydantic import BaseModel
from env import get_env_var
from typing import Optional
import ollama

class ModelResponseStructure(BaseModel):
    """
    Model response structure.
    """
    topic_title: str                                # The title of the model's response
    response: str                                   # The model's response to the query
    followup_questions: Optional[list[str]] = None  # Relevant follow-up questions to ask the patient. Populate this field if you feel clarifying questions are needed

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

    response_obj = ModelResponseStructure.model_validate_json(response.message.content)

    print(f"Model response: {response_obj}")
    
    return (response_obj.topic_title,
            response_obj.response, 
            response_obj.followup_questions)