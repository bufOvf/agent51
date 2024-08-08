import os
from dotenv import load_dotenv
from groq_backend import GroqBackend


load_dotenv()
def main():
    # Get Groq API key from environment variable
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")


    user_name = input("Your Username. Press enter for default: ").lower()
    if not user_name:
        user_name = os.environ.get('default_user_name')
    conversation_length = 0

    # Initialize GroqBackend
    backend = GroqBackend(user_name, api_key=groq_api_key)

    print("Mira is initialized and ready to chat. Type your messages. Type 'exit' to end the conversation.")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            response = backend.get_response(user_input)
            print(f"Mira: {response}")

            conversation_length += 1

    except KeyboardInterrupt:
        print("\nConversation ended by user interruption.")

    print("Mira's going offline now. The conversation has been saved!")

if __name__ == "__main__":
    main()