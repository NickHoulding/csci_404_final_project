"""
Handles Streamlit chat interface logic.
"""

# Imports
from rag import get_context_prompt, load_kb, get_embedding
from models import query_model
from env import get_env_var
from utils import load_css
import streamlit as st
import os

# Globals
kb = load_kb(os.path.join(
    os.path.dirname(__file__), 
    'knowledge', 
    get_env_var('KNOWLEDGE_BASE')
))

def initialize_session_state():
    # Initialize session state variables if they don't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    # Displays the chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message(role: str, content: str):
    # Add a message to the chat history
    st.session_state.messages.append({
        "role": role, 
        "content": content
    })

def run_chat_interface():
    # Set up the Streamlit page configuration
    load_css()
    initialize_session_state()
    display_chat_history()
    
    # Wait for user input
    if user_prompt := st.chat_input("What are the patient's symptoms?"):
        add_message("user", user_prompt)
        
        # Display spinner and process input
        with st.spinner("Thinking..."):
            # Embed the user input
            prompt_embedding = get_embedding(user_prompt)
            
            # Search the knowledge base for most relevant context
            results = kb.search(q_embed=prompt_embedding, top_k=3)

            # Add retrieved context to the user prompt
            context_prompt = get_context_prompt(user_prompt, results)
            
            # Generate a response using the decoding model
            title, response = query_model(context_prompt)
            output = f"### {title}\n{response}\n\n*{get_env_var('DISCLAIMER')}*"
        
        add_message("assistant", output)
        st.rerun()

# Entry point
if __name__ == "__main__":
    run_chat_interface()