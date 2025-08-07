from chatbot_response import chatbot_response

print("Bot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chatbot_response(user_input)
    print("Bot:", response)
