import json
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import Counter

# Load the filtered tokens from the JSON file
with open('filtered_tokens_1000.json', 'r', encoding='utf-8') as file:
    filtered_tokens_list = json.load(file)


# Join tokens into strings
documents = [' '.join(map(str, tokens)) for tokens in filtered_tokens_list]

from sentence_transformers import SentenceTransformer

# Load a pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the documents
embeddings = model.encode(documents, show_progress_bar=True)

from sklearn.cluster import KMeans

# Define the number of clusters
num_clusters = 5  # Adjust based on your dataset

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(embeddings)

# Assign cluster labels to documents
cluster_labels = kmeans.labels_

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Vectorize the documents
vectorizer = TfidfVectorizer(stop_words='english')
X_matrice = vectorizer.fit_transform(documents)
print(X_matrice)

# Get the terms corresponding to the features
terms = vectorizer.get_feature_names_out()

# Find the top terms in each cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print(f"Cluster {i + 1}:")
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]  # Top 10 terms
    print(", ".join(top_terms))
    print()

pca = PCA(n_components=0.50, random_state=42)  # Retain 95% of variance
reduced_embeddings = pca.fit_transform(embeddings)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean', n_jobs=-1)
cluster_labels = dbscan.fit_predict(reduced_embeddings)

# Analyze and print top words per cluster
terms = vectorizer.get_feature_names_out()
clustered_docs = {i: [] for i in set(cluster_labels) if i != -1}  # Exclude noise (-1)

for doc_id, cluster_id in enumerate(cluster_labels):
    if cluster_id != -1:
        clustered_docs[cluster_id].append(doc_id)

# Find the top terms in each cluster
for cluster_id, doc_ids in clustered_docs.items():
    word_freq = np.sum(X_matrice[doc_ids], axis=0)
    top_terms = [terms[ind] for ind in word_freq.argsort()[0, -10:][::-1]]  # Top 10 terms
    print(f"Cluster {cluster_id}:")
    print(", ".join(map(str, top_terms)))
