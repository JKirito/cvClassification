import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import re
import json

# Huggingface login
HF_AUTH_TOKEN = "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"
login(token=HF_AUTH_TOKEN)

# Set the model name and local directory
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
local_model_path = "./mistral_model"

# Create the data folder if it doesn't exist
data_folder = "./data"
os.makedirs(data_folder, exist_ok=True)

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Download and save the model and tokenizer if not already present
if not os.path.exists(local_model_path):
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_AUTH_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=HF_AUTH_TOKEN,
        quantization_config=quantization_config
    )
    
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
    print("Model downloaded and saved.")
else:
    print("Loading model from local directory...")

# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Function to tokenize string
def tokenize_string(text: str):
    return re.findall(r"\w+|[^\w\s]", text)

# Function to check for duplicates
def is_duplicate(new_cv, existing_cvs):
    return any(all(new_cv[i] == existing_cv[i] for i in range(len(new_cv))) for existing_cv in existing_cvs)

# Function to flatten a nested list
def flatten_list(nested_list):
    flattened = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened.extend(flatten_list(sublist))
        else:
            flattened.append(sublist)
    return flattened

# Global lists to store generated CVs and tokenized versions
CV_data = []
tokenized_CV_data = []

# Chat history to track successful generations
chat_history = []

# Counter for successful CV generations
successful_cv_count = 0

# Generate 5 CVs
while successful_cv_count < 5:
    i = successful_cv_count

    # Clean the CV text to remove the initial prompt and unrelated content
    prompt = """    
    Generate CV with simple design containing 5 sections 'About Me', 'Education', 'Skills', 'Work Experience', and 'Achievements'.
    Ensure all names and dates are randomly generated and realistic.
    Do not use any placeholders or instructions.
    """

    cv_generation_conversation = [{
            "role": "user",
            "content": prompt
    }]
    # Generate input tokens using the updated conversation with chat history
    inputs = tokenizer.apply_chat_template(cv_generation_conversation, add_generation_prompt=True, return_tensors="pt")
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    inputs = inputs.to(model.device)
    attention_mask = attention_mask.to(model.device)

    # Generate CV output with increased randomness
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        repetition_penalty=1.2
    )

    cv_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean the CV text to remove the initial prompt and unrelated content
    clean_text = cv_text.replace(prompt, "").strip()
    
    print(f"\n----- CV {i + 1} ------")

    # Tokenize each section of the CV
    tokenized_cv = [tokenize_string(section) for section in clean_text.split("\n")]

    # Ensure no duplicates before storing the CV
    if not is_duplicate(tokenized_cv, tokenized_CV_data):
        # Append the cleaned CV to CV_data
        CV_data.append(clean_text)
        
        # Append the tokenized CV to tokenized_CV_data
        tokenized_CV_data.append(tokenized_cv)
        
        # Update the successful CV count
        successful_cv_count += 1
        
        # Append the current conversation (prompt and response) to chat history
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": clean_text})
        
        print(clean_text)
        print(f"Total CVs generated: {successful_cv_count}")
        print(f"Total tokenized CVs generated: {len(tokenized_CV_data)}")
    else:
        print("Duplicate or invalid CV generated, retrying...")
        print(clean_text)
        continue

# Flatten the tokenized CV data
flattened_tokenized_CV_data = [flatten_list(cv) for cv in tokenized_CV_data]

# Save the CV data and flattened tokenized CV data as JSON files
cv_data_path = os.path.join(data_folder, "CV_data.json")
tokenized_cv_data_path = os.path.join(data_folder, "tokenized_CV_data.json")
chat_history_path = os.path.join(data_folder, "chat_history.json")

with open(cv_data_path, 'w') as cv_file:
    json.dump(CV_data, cv_file, indent=4)

with open(tokenized_cv_data_path, 'w') as tokenized_cv_file:
    json.dump(flattened_tokenized_CV_data, tokenized_cv_file, indent=4)

# Save the chat history
with open(chat_history_path, 'w') as chat_file:
    json.dump(chat_history, chat_file, indent=4)

print("CV generation complete.")
