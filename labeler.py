import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

# Set the model name and local directory
model_name = "deepseek-ai/deepseek-llm-7b-chat"
local_model_path = "./deepseek"

# Create the data folder if it doesn't exist
data_folder = "./data"
os.makedirs(data_folder, exist_ok=True)

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model and tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    quantization_config=quantization_config
)

# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as pad token
    model.config.pad_token_id = tokenizer.pad_token_id

# Load the tokenized CV data
tokenized_cv_data_path = os.path.join(data_folder, "tokenized_CV_data.json")
with open(tokenized_cv_data_path, 'r') as file:
    tokenized_CV_data = json.load(file)

def generate_labels(tokens):
    # Define the prompt for labeling
    prompt_template = f"""
You are a CV labeling assistant. Your task is to assign labels to each token in the given CV text according to the provided schema. 

Tokens: {tokens}

Label each token using the following schema:

0: O (Other/Outside any entity)
1: B-ABOUT (Beginning of About section)
2: I-ABOUT (Inside About section)
3: B-SKILL_TITLE (Beginning of Skill title)
4: I-SKILL_TITLE (Inside Skill title)
5: I-SKILL_PROFICIENCY (Skill proficiency level)
6: B-LANGUAGE_TITLE (Beginning of Language title)
7: I-LANGUAGE_TITLE (Inside Language title)
8: I-LANGUAGE_PROFICIENCY (Language proficiency level)
9: B-EXPERIENCE_TITLE (Beginning of Experience title)
10: I-EXPERIENCE_WORK_AT (Company/Organization worked at)
11: I-EXPERIENCE_START_DATE (Start date of experience)
12: I-EXPERIENCE_END_DATE (End date of experience)
13: B-EDUCATION_TITLE (Beginning of Education title)
14: I-EDUCATION_INSTITUTION (Educational institution)
15: I-EDUCATION_START_DATE (Start date of education)
16: I-EDUCATION_END_DATE (End date of education)
17: I-EDUCATION_MAJOR (Major or field of study)
18: B-ACHIEVEMENT_TITLE (Beginning of Achievement title)
19: I-ACHIEVEMENT_CONTENT (Content of achievement)
20: I-ACHIEVEMENT_DATE (Date of achievement)

Instructions:
1. Carefully analyze each token and its context within the CV.
2. Assign the most appropriate label to each token based on its role and position in the CV.
3. Use the B- prefix for the first token of an entity and I- for subsequent tokens of the same entity.
4. Use 0 for tokens that don't belong to any specific entity or section.
5. Pay close attention to section headers, dates, company names, and skill descriptions.
6. Ensure that your labeling is consistent and follows the logical structure of a CV.

Example:
Tokens: ["John", "Doe", "Software", "Engineer", "About", "Me", "Experienced", "developer", "with", "5", "years", "of", "experience"]
Labels: [0, 0, 3, 4, 1, 2, 2, 2, 2, 2, 2, 2, 2]

Now, provide the labels for the given tokens. Your output should be a Python list of integers, with exactly one label per token. Do not include any explanations or additional text in your response, only the list of labels.
"""

    prompt = prompt_template
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=10000,
        temperature=0.7,  # Add some randomness to avoid repetitive outputs
        top_p=0.95,  # Use nucleus sampling
        do_sample=True  # Enable sampling
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Print the raw response for debugging
    print("Model response:", response)
    
    # Extract labels
    try:
        # Find the last occurrence of '[' and the first occurrence of ']' after that
        start_index = response.rfind('[')
        end_index = response.find(']', start_index)
        if start_index != -1 and end_index != -1:
            labels_str = response[start_index:end_index+1]
            labels = [int(label) for label in labels_str.strip("[]").split(",") if label.strip()]
            return labels
        else:
            print("No valid list found in the response")
            return []
    except ValueError as e:
        print("Error processing labels:", e)
        return []


# Process each tokenized CV
labeled_CV_data = []
for i, tokenized_cv in enumerate(tokenized_CV_data):
    # Flatten the tokenized CV for labeling
    flattened_tokens = [token for sublist in tokenized_cv for token in sublist]
    labels = generate_labels(flattened_tokens)
    
    # Create the result dictionary
    result = {
        'id': str(i),
        'ner_tags': labels,
        'tokens': flattened_tokens
    }
    labeled_CV_data.append(result)

# Save the labeled CV data as a JSON file
labeled_cv_data_path = os.path.join(data_folder, "labeled_CV_data.json")
with open(labeled_cv_data_path, 'w') as file:
    json.dump(labeled_CV_data, file, indent=4)

print("Labeled CV data saved to './data/labeled_CV_data.json'")
