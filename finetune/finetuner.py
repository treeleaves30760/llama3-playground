import json
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from datasets import Dataset


class LlamaFineTuner:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cuda"):
        self.device = device
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            model_name).to(self.device)

    def load_data(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        return Dataset.from_dict({"text": data})

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    def fine_tune(self, train_data, output_dir="./fine_tuned_model", num_train_epochs=3, per_device_train_batch_size=8):
        tokenized_data = train_data.map(self.tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=1000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data,
        )

        trainer.train()
        trainer.save_model()

    def generate_text(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    fine_tuner = LlamaFineTuner()
    train_data = fine_tuner.load_data("training_data.json")
    fine_tuner.fine_tune(train_data)

    # Example of generating text with the fine-tuned model
    generated_text = fine_tuner.generate_text("Once upon a time")
    print(generated_text)
