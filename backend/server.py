from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Configure CORS to allow all origins from localhost
CORS(app, resources={
     r"/*": {"origins": ["http://localhost:*", "https://localhost:*"]}})

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model(model_name, local=False):
    global model, tokenizer
    if local:
        # Load model from local path
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True)
    else:
        # Load model from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)


@app.route('/load_model', methods=['POST'])
def load_model_route():
    data = request.json
    model_name = data.get('model_name')
    local = data.get('local', False)

    try:
        load_model(model_name, local)
        return jsonify({"message": f"Model {model_name} loaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/submit', methods=['POST'])
def submit():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not loaded. Please load a model first."}), 400

    data = request.json
    prompt = data.get('prompt')

    # New parameters
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 40)
    top_p = data.get('top_p', 0.95)
    min_p = data.get('min_p', 0.05)
    n_predict = data.get('n_predict', -1)
    stop = data.get('stop', [])
    repeat_penalty = data.get('repeat_penalty', 1.1)
    repeat_last_n = data.get('repeat_last_n', 64)
    penalize_nl = data.get('penalize_nl', True)
    presence_penalty = data.get('presence_penalty', 0.0)
    frequency_penalty = data.get('frequency_penalty', 0.0)
    mirostat = data.get('mirostat', 0)
    mirostat_tau = data.get('mirostat_tau', 5.0)
    mirostat_eta = data.get('mirostat_eta', 0.1)
    seed = data.get('seed', -1)
    ignore_eos = data.get('ignore_eos', False)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)

        # Prepare stop sequences
        stop_sequences = [tokenizer.encode(s, return_tensors="pt")[
            0] for s in stop]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=input_ids.shape[1] +
                (n_predict if n_predict > 0 else 1000),
                num_return_sequences=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repeat_penalty,
                no_repeat_ngram_size=repeat_last_n,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=None if ignore_eos else tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Apply stop sequences
        for stop_seq in stop:
            if stop_seq in response:
                response = response[:response.index(stop_seq)]

        return jsonify({"response": response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
