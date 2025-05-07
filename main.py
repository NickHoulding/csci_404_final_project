"""
MediChat: A medical AI chatbot.
"""

# Imports
import subprocess
import os

def main():
    """
    Launches the Streamlit interface for MediChat
    """
    file_path = os.path.join(
        os.path.dirname(__file__),
        'chat_gui.py'
    )
    
    try:
        subprocess.run(
            ["streamlit", "run", file_path], 
            check=True
        )
    
    except FileNotFoundError:
        print("""Error: Streamlit not found. 
              Install it using 'pip install streamlit'""")
    
    except subprocess.CalledProcessError:
        print("Error: Failed to start Streamlit interface")

# Entry point
if __name__ == "__main__":
    main()