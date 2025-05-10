"""
Handles environment variables for the application.
"""

# Imports
from dotenv import load_dotenv
from typing import Optional
import os

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), ".env"),
)

def get_env_var(key: str) -> Optional[str]:
    """
    Get the value of an environment variable.

    Args:
        key (str): The name of the environment variable.

    Returns:
        Optional[str]: The value of the environment variable, 
            or None if not found.
    """
    return os.getenv(key)