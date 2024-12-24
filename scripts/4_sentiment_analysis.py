import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset


ratings = []
# Load the data
with open('filtered_review_200.jsonl', 'r', encoding='utf-8') as file:
    reviews = [json.loads(line) for line in file]
    for review in reviews:
        ratings.append(review['rating'])

# Combine title and text for each review
documents = [f"{review['title']} {review['text']}" for review in reviews]

# Specify the model name
model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize and encode the texts
inputs = tokenizer(
    documents,
    return_tensors='pt',
    padding=True,
    truncation=True
)

# Ensure the model is in evaluation mode
model.eval()
# Define a custom dataset
class ReviewDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Create a DataLoader with a suitable batch size
batch_size = 8  # Adjust based on your system's memory capacity
dataset = ReviewDataset(documents)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define sentiment labels
sentiment_labels = ['1 star', '2 stars', '3 stars', '4 stars', '5 stars']

# Process each batch
all_predicted_sentiments = []
for batch in dataloader:
    # Tokenize and encode the batch
    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)

    # Perform a forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Obtain the logits and apply softmax to get probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class indices
    predicted_class_indices = torch.argmax(probabilities, dim=1)

    # Map indices to labels
    predicted_sentiments = [sentiment_labels[idx] for idx in predicted_class_indices]
    all_predicted_sentiments.extend(predicted_sentiments)

# Display the results for the first 10 reviews
for i, (text, sentiment) in enumerate(zip(documents, all_predicted_sentiments)):
    if i >= 10:
        break
    print(f"Review {i + 1}:")
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")

# Define a mapping from sentiment labels to numerical scores
sentiment_to_score = {
    '1 star': 1,
    '2 stars': 2,
    '3 stars': 3,
    '4 stars': 4,
    '5 stars': 5
}

# Convert predicted sentiments to numerical scores
predicted_scores = [sentiment_to_score[sentiment] for sentiment in all_predicted_sentiments]


import numpy as np
from scipy.stats import pearsonr

# Convert lists to numpy arrays
actual_ratings = np.array(ratings)
predicted_scores = np.array(predicted_scores)

# Calculate Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(actual_ratings, predicted_scores)

print(f"Pearson Correlation Coefficient: {correlation_coefficient:.2f}")
print(f"P-value: {p_value:.2e}")