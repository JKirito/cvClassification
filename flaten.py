import json

# Load the data from the JSON file
with open('./data/tokenized_cv_data.json', 'r') as file:
    data = json.load(file)

# Function to flatten the nested list
def flatten_list(nested_list):
    flattened = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened.extend(flatten_list(sublist))
        else:
            flattened.append(sublist)
    return flattened

# Flatten each element in the main list
flattened_data = [flatten_list(item) for item in data]

# Save the flattened data to a new file
with open('./data/flattened_cv_data.json', 'w') as file:
    json.dump(flattened_data, file, indent=2)

print("Flattened data saved to './data/flattened_cv_data.json'")
