# Import necessary libraries
import json
from rouge import Rouge

# load responses_answers.json 
def load_responses(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
# Path to the JSON file
responses_file_path = 'responses_answers.json'

# Load the JSON file with the responses and references
data = load_responses(responses_file_path)

# Extract hypotheses and references from the JSON data
hyps = [item['hyp'] for item in data]
refs = [item['ref'] for item in data]

# Initialize the Rouge object
rouge = Rouge()

# Calculate average ROUGE scores for ROUGE-1, ROUGE-2, and ROUGE-L
scores = rouge.get_scores(hyps, refs, avg=True)

# Print the average ROUGE scores
#print("Average ROUGE Scores:")
#print(json.dumps(scores, ensure_ascii=False, indent=4))

# Save results to a JSON file

# Path of JSON file for saving results
save_file_path = 'rouge_scores.json'

with open(save_file_path, 'w', encoding='utf-8') as f:
    json.dump(scores, f, ensure_ascii=False, indent=4)

print(f"ROUGE scores have been saved to {save_file_path}")