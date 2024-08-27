import requests
import json


class LLMInteraction:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.history = []

    def load_model(self, model_name, local=False):
        url = f"{self.server_url}/load_model"
        data = {
            "model_name": model_name,
            "local": local
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Model {model_name} loaded successfully")
        else:
            print(f"Error loading model: {response.json().get('error')}")

    def ask(self, question, **kwargs):
        url = f"{self.server_url}/submit"

        # Prepare the prompt with history
        full_prompt = self._prepare_prompt(question)

        data = {
            "prompt": full_prompt,
            **kwargs
        }

        response = requests.post(url, json=data)

        if response.status_code == 200:
            answer = response.json().get('response')
            # Add the new question and answer to the history
            self.history.append({"question": question, "answer": answer})
            return answer
        else:
            return f"Error: {response.json().get('error')}"

    def _prepare_prompt(self, new_question):
        prompt = "Here's the conversation history:\n\n"
        for item in self.history:
            prompt += f"Human: {item['question']}\nAI: {item['answer']}\n\n"
        prompt += f"Human: {new_question}\nAI:"
        return prompt

    def clear_history(self):
        self.history = []
        print("Conversation history cleared.")

    def show_history(self):
        for i, item in enumerate(self.history, 1):
            print(f"--- Turn {i} ---")
            print(f"Human: {item['question']}")
            print(f"AI: {item['answer']}")
            print()


# Example usage:
if __name__ == "__main__":
    llm = LLMInteraction()

    # Load a model (adjust the model name as needed)
    llm.load_model("gpt2")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        elif user_input.lower() == 'clear history':
            llm.clear_history()
        elif user_input.lower() == 'show history':
            llm.show_history()
        else:
            response = llm.ask(user_input)
            print("AI:", response)
