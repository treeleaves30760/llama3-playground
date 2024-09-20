import random
from interact import LLMInteraction


class CasualChatBot:
    def __init__(self, model_name="gpt2", server_url="http://localhost:5000"):
        self.llm = LLMInteraction(server_url)
        self.llm.load_model(model_name)
        self.greetings = ["Hello!", "Hi there!", "Hey!", "Greetings!"]
        self.farewells = ["Goodbye!", "See you later!", "Bye!", "Take care!"]
        self.conversation_starters = [
            "How's your day going?",
            "Got any exciting plans?",
            "What's on your mind?",
            "Tell me something interesting!",
        ]

    def greet(self):
        return random.choice(self.greetings)

    def farewell(self):
        return random.choice(self.farewells)

    def start_conversation(self):
        return random.choice(self.conversation_starters)

    def respond(self, user_input):
        response = self.llm.ask(user_input,
                                temperature=0.7,
                                max_length=100,
                                top_p=0.9)
        return response

    def chat(self):
        print("ChatBot:", self.greet())
        print("ChatBot:", self.start_conversation())

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("ChatBot:", self.farewell())
                break
            elif user_input.lower() == 'clear history':
                self.llm.clear_history()
                print("ChatBot: Conversation history cleared. Let's start fresh!")
            elif user_input.lower() == 'show history':
                self.llm.show_history()
            else:
                response = self.respond(user_input)
                print("ChatBot:", response)


if __name__ == "__main__":
    chatbot = CasualChatBot()
    chatbot.chat()
