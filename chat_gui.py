"""
Handles Streamlit chat interface logic.
"""

from models import query_model
from env import get_env_var
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
    st.set_page_config(
        page_title="MediChat",
        page_icon="🏥",
        layout="centered"
    )
    
    # CSS for custom styling
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
            font-size: 3rem;
            margin-bottom: 1rem;
            font-weight: bold;
            color: #4287f5;
            text-shadow: 0px 0px 10px rgba(66, 135, 245, 0.3);
        }
        
        .stTextInput>div>div>input {
            border-radius: 10px;
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4287f5;
        }
        
        .stChatMessage {
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .stChatMessage[data-testid="user-message"] {
            background-color: #3a3f4b !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #262730 !important;
        }
        
        .stSpinner>div {
            border-top-color: #4287f5 !important;
        }
        
        em {
            color: #a1a1a1;
            font-size: 0.9rem;
        }
        
        .stChatInputContainer {
            background-color: #1e232a;
            border-top: 1px solid #333;
            padding-top: 10px;
        }
        </style>""", 
        unsafe_allow_html=True)
    
    st.markdown(
        "<h1 class='centered-title'>MediChat</h1>", 
        unsafe_allow_html=True
    )
    
    initialize_session_state()
    display_chat_history()
    
    if prompt := st.chat_input("What are the patient's symptoms?"):
        add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_model(prompt)
                formatted_response = f"{response}\n\n*{get_env_var('DISCLAIMER')}*"
                st.markdown(formatted_response)
        
        add_message("assistant", formatted_response)

# Entry point
if __name__ == "__main__":
    run_chat_interface()