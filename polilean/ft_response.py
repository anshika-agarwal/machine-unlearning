import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Define paths based on your repo structure
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # Ensure you have access to this model
ADAPTER_PATH = "./llama_conservative_ft"  # Path to your fine-tuned adapter
STATEMENT_FILE = "./polilean/response/example.jsonl"  # Input file containing prompts
OUTPUT_FILE = "./polilean/response/FT-Llama-3.2-1B.jsonl"  # Output file to store responses

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Load model and tokenizer
def load_model(model_name, adapter_path, device):
    """Loads the base model and adapter model on the specified device."""
    base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model.to(device), tokenizer

# Generate response for each statement
def generate_response(model, tokenizer, statement, device, max_new_tokens=100):
    """Generates a response for a given statement using the model."""
    prompt = f"Please respond to the following statement: {statement}\nYour response:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

# Main execution
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model(MODEL_NAME, ADAPTER_PATH, device)

# Load statements
with open(STATEMENT_FILE, "r") as f:
    statements = json.load(f)

# Generate responses
for item in tqdm(statements, desc="Generating responses"):
    item["response"] = generate_response(model, tokenizer, item["statement"], device)

# Save responses in polilean/response/
with open(OUTPUT_FILE, "w") as f:
    json.dump(statements, f, indent=4)

print(f"Responses saved at: {OUTPUT_FILE}")
