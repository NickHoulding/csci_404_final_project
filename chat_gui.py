"""
Handles Streamlit chat interface logic.
"""

from models import query_model
from rag import search_top_k
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
    
    if prompt := st.chat_input("What are the patient's symptoms?"):
        add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results = search_top_k(query_text=prompt, k=3)

                # TODO: Add instructions to tell the AI how to use the context in its response.
                contexts = " ".join([x["text"] for x in results])
                
                title, response, followup_questions = query_model(contexts)
                output_response = f"### {title}\n{response}\n"

                if followup_questions:
                    output_response += "##### Follow-up Questions\n"
                    for question in followup_questions:
                        output_response += f"- {question}\n"
        
                output_response += f"\n*{get_env_var('DISCLAIMER')}*"
                st.markdown(output_response)

        add_message("assistant", output_response)

# Entry point
if __name__ == "__main__":
    run_chat_interface()