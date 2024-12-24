#First virtual environment 
#python -m venv classmodules
#source classmodules/bin/activate
#python -m spacy download en_core_web_sm

import re
import json
import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

combined_reviews = []
filtered_tokens_list = []
# Initialize lists to store extracted data
texts = []
ratings = []
titles = []

# Read and parse the .jsonl file
with open('filtered_review_1000.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        review = json.loads(line)
        texts.append(review['text'])
        ratings.append(review['rating'])
        titles.append(review['title'])
        # Combine title and text with a space separator
        combined_text = f"{review['title']} {review['text']}"
        combined_reviews.append(combined_text)
print(len(combined_reviews))
        #print(review['text'])

combined_reviews = [review for review in combined_reviews if isinstance(review, str)]
print(len(combined_reviews))

# Process the list of texts
docs = list(nlp.pipe(combined_reviews))

# Process the list of texts
for doc in nlp.pipe(combined_reviews):
    # Filter out stopwords and punctuation, and lemmatize tokens
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    filtered_tokens_list.append(filtered_tokens)

print("Filtered Tokens:", filtered_tokens_list)
print("Length of Filtered Tokens:", len(filtered_tokens_list))

# Specify the filename
output_filename = 'filtered_tokens_1000.json'

# Open the file in write mode and save the list
with open(output_filename, 'w', encoding='utf-8') as file:
    json.dump(filtered_tokens_list, file, ensure_ascii=False, indent=4)

print(f"Filtered tokens have been saved to {output_filename}")