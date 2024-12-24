import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from collections import Counter

# Load the filtered tokens from the JSON file
with open('filtered_tokens_1000.json', 'r', encoding='utf-8') as file:
    filtered_tokens_list = json.load(file)


# Join tokens into strings
documents = [' '.join(tokens) for tokens in filtered_tokens_list]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = vectorizer.fit_transform(documents)
print("tf_idf_matrix",tfidf_matrix)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
print(feature_names)

# Convert TF-IDF matrix to a dense format
dense_tfidf = tfidf_matrix.todense()
print(dense_tfidf)

# Create a list of dictionaries to hold term and score information
tfidf_scores = []
for doc_idx, doc in enumerate(dense_tfidf):
    term_scores = {}
    for term_idx, score in enumerate(doc.tolist()[0]):
        if score > 0:
            term_scores[feature_names[term_idx]] = score
    tfidf_scores.append(term_scores)
#print(tfidf_scores)

# Print TF-IDF scores for the first document
print("TF-IDF scores for the first document:")
#for term, score in tfidf_scores[0].items():
#    print(f"Term: {term}, Score: {score}")


# Define the number of clusters (topics)
num_clusters = 5  # Adjust based on your dataset

# Initialize and fit KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

# Get cluster centers and terms
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()

# Display the top terms for each cluster
for i in range(num_clusters):
    #print(f"Topic {i + 1}:")
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]  # Top 10 terms
    #print(", ".join(top_terms))
    #print()

#use DBSCAN
#use Elbow or Silhouette
#noise reduction
#reduce dimensionality

from sklearn.metrics import silhouette_score

# Define the range of clusters to evaluate
cluster_range = range(2, 11)  # Silhouette score is undefined for k=1
silhouette_scores = []

# Calculate silhouette scores for each number of clusters
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the Silhouette Analysis graph
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Silhouette Score')
plt.title('Silhouette Analysis for Optimal Number of Clusters')
plt.xticks(cluster_range)
plt.grid(True)
#plt.show()

# Reduce dimensionality with PCA
#pca = PCA(n_components=0.50, random_state=42)  # Retain 95% of variance
#reduced_data = pca.fit_transform(tfidf_matrix.toarray())

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean', n_jobs=-1)
clusters = dbscan.fit_predict(tfidf_matrix)

# Analyze and print top words per cluster
terms = vectorizer.get_feature_names_out()
clustered_docs = {i: [] for i in set(clusters) if i != -1}  # Exclude noise (-1)

for doc_id, cluster_id in enumerate(clusters):
    if cluster_id != -1:
        clustered_docs[cluster_id].append(doc_id)

for cluster_id, doc_ids in clustered_docs.items():
    word_freq = Counter()
    for doc_id in doc_ids:
        word_freq.update(documents[doc_id].split())
    top_words = word_freq.most_common(10)
    #print(f"Cluster {cluster_id}:")
    #for word, freq in top_words:
    #    print(f"  {word}: {freq}")
