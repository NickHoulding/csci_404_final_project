"""
Handles Streamlit chat interface logic.
"""

from rag import get_context_prompt, get_embedding, search_kb
from models import query_model
from env import get_env_var
from utils import load_css
import streamlit as st

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_chat_history():
    """Displays the chat history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message(role, content):
    """Add a message to the chat history"""
    st.session_state.messages.append({
        "role": role, 
        "content": content
    })

def run_chat_interface():
    """Main function to run the Streamlit chat interface"""
    load_css()
    
    st.markdown(
        "<h1 class='centered-title'>MediChat</h1>", 
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    display_chat_history()
    
    if user_prompt := st.chat_input("What are the patient's symptoms?"):
        add_message("user", user_prompt)
        
        with st.spinner("Thinking..."):
            prompt_embedding = get_embedding(user_prompt)
            results = search_kb(prompt_embedding, top_k=3)
            context_prompt = get_context_prompt(user_prompt, results)
            title, response = query_model(context_prompt)
            output = f"### {title}\n{response}\n\n*{get_env_var('DISCLAIMER')}*"
        
        add_message("assistant", output)
        st.rerun()

# Entry point
if __name__ == "__main__":
    run_chat_interface()