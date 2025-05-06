"""
MediChat: A medical AI chatbot.
"""

# Imports
from models import query_model

# Globals
DISCLAIMER = "This is not a diagnosis; consult a physician."

def main():
    print("MediChat: A medical AI chatbot.\nType 'quit' to quit.\n")

    is_active = True
    try:
        while is_active:
            user_input = input("User: ").strip().lower()
            
            if user_input == "quit":
                is_active = False
                print("\nExiting the chat. Goodbye!")
            elif user_input:
                response = query_model(user_input)
                print(f"\nMediChat: {response}\n\n{DISCLAIMER}\n")
    
    except KeyboardInterrupt:
        print("\n\nExiting the chat. Goodbye!")

# Entry point
if __name__ == "__main__":
    main()