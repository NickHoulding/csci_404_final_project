"""
Medical AI chatbot for respiratory insights.
"""

# Imports
import subprocess
import os

def main():
    # Locate the chat GUI script
    file_path = os.path.join(
        os.path.dirname(__file__),
        'chat_gui.py'
    )
    
    # Run the Streamlit interface
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

    except KeyboardInterrupt:
        print("\nExit: Stopped by user")

# Entry point
if __name__ == "__main__":
    main()