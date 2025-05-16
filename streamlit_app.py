"""
Handles Streamlit app configuration and startup.
"""

from chat_gui import run_chat_interface
from utils import load_css
import streamlit as st

# Chat page configuration
st.set_page_config(
    page_title="MediChat",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="none"
)

# Load custom CSS
load_css()

run_chat_interface()