# LLama 3 Playground

This project provides a comprehensive playground for experimenting with and fine-tuning LLama 3 models. It includes a backend server, an interaction class for easy communication with the server, and a light fine-tuning framework.

## Features

1. LLama 3 Backend Server
2. LLMInteraction Class for easy interaction with the webserver or backend
3. Light Fine-tuning Framework

## Installation

To set up the LLama 3 Playground, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/treeleaves30760/llama3-playground
   cd llama3-playground
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Components

### 1. LLama 3 Backend Server

The backend server is built using Flask and provides an API for loading models and generating text. It supports various parameters for text generation and can load models both locally and from Hugging Face.

Key features:

- Load models from local storage or Hugging Face
- Generate text with customizable parameters
- CORS support for localhost origins

To run the server:

```bash
python server.py
```

The server will start on `http://localhost:5000`.

#### API Endpoints

- `POST /load_model`: Load a model
  - Parameters:
    - `model_name`: Name of the model to load
    - `local` (optional): Boolean to indicate if the model should be loaded from local storage

- `POST /submit`: Generate text
  - Parameters:
    - `prompt`: The input text prompt
    - Various generation parameters (temperature, top_k, top_p, etc.)

### 2. LLMInteraction Class

This class provides an easy-to-use interface for interacting with the LLama 3 backend server. It maintains conversation history and allows for seamless communication with the model.

Key features:

- Load models
- Ask questions and get responses
- Maintain conversation history
- Clear and display conversation history

Example usage:

```python
from llm_interaction import LLMInteraction

llm = LLMInteraction()
llm.load_model("gpt2")

response = llm.ask("What is the capital of France?")
print("AI:", response)

llm.show_history()
```

### 3. Fine-tuning Framework

The `LlamaFineTuner` class provides a lightweight framework for fine-tuning LLama models on custom datasets.

Key features:

- Load and preprocess data from JSON files
- Fine-tune models with customizable training arguments
- Generate text using the fine-tuned model
- Estimate perplexity of generated text

Example usage:

```python
from llama_finetuner import LlamaFineTuner

fine_tuner = LlamaFineTuner()
train_data = fine_tuner.load_data("training_data.json")

train_result, eval_result = fine_tuner.fine_tune(train_data)

generated_text = fine_tuner.generate_text("Once upon a time")
perplexity = fine_tuner.estimate_perplexity(generated_text)
```

## Usage

1. Start the backend server:

   ```bash
   python server.py
   ```

2. In a separate terminal, run the interaction script or import the `LLMInteraction` class in your own script:

   ```bash
   python llm_interaction.py
   ```

3. To fine-tune a model, use the `LlamaFineTuner` class as shown in the example above.

## Contributing

Contributions to the LLama 3 Playground are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the Hugging Face Transformers library.
- Thanks to the LLama team for their work on the LLama language model.
