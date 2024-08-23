import json
import torch
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split


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

    def fine_tune(self, train_data, output_dir="./fine_tuned_model", num_train_epochs=3, per_device_train_batch_size=8, validation_split=0.1):
        # Split the data into train and validation sets
        train_val = train_data.train_test_split(test_size=validation_split)
        train_data = train_val['train']
        val_data = train_val['test']

        tokenized_train = train_data.map(self.tokenize_function, batched=True)
        tokenized_val = val_data.map(self.tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=1000,
            save_total_limit=2,
            evaluation_strategy="epoch",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
        )

        train_result = trainer.train()
        trainer.save_model()

        # Evaluate the model
        eval_result = trainer.evaluate()

        return train_result, eval_result

    def generate_text(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def estimate_perplexity(self, text):
        encodings = self.tokenizer(text, return_tensors="pt").to(self.device)
        max_length = min(
            encodings.input_ids.shape[1], self.model.config.max_position_embeddings)
        stride = 512
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:,
                                            begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()


if __name__ == "__main__":
    fine_tuner = LlamaFineTuner()
    train_data = fine_tuner.load_data("training_data.json")

    # Fine-tune the model and get training and evaluation results
    train_result, eval_result = fine_tuner.fine_tune(train_data)

    print("Training results:", train_result)
    print("Evaluation results:", eval_result)

    # Example of generating text with the fine-tuned model
    generated_text = fine_tuner.generate_text("Once upon a time")
    print("Generated text:", generated_text)

    # Estimate perplexity of the generated text
    perplexity = fine_tuner.estimate_perplexity(generated_text)
    print(f"Estimated perplexity: {perplexity}")
