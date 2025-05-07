"""
Handles Streamlit app configuration and startup.
"""

import streamlit as st
from chat_gui import run_chat_interface

# Chat page configuration
st.set_page_config(
    page_title="MediChat",
    layout="wide",
    initial_sidebar_state="none"
)

# Some custom CSS for improved appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

run_chat_interface()