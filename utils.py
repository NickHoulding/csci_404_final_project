"""
Utility functions for the application.
"""

import streamlit as st
import os

def load_css():
    """Load custom CSS styling from the .streamlit/style.css file"""
    css_file_path = os.path.join(
        os.path.dirname(__file__), 
        '.streamlit', 
        'style.css'
    )
    
    with open(css_file_path, 'r') as f:
        css = f.read()
    
    st.markdown(f"<style>{css}</style>", 
                unsafe_allow_html=True)