#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=UserWarning, 
                       module="joblib.externals.loky.backend.context")

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

possible_paths = [
    "ml_corpuses.csv",
]

df = None

for path in possible_paths:
    if os.path.exists(path):
        print(f"File found: {path}")
        try:
            df = pd.read_csv(path, encoding="utf-8")
            print(f"Successfully read file: {path}")
            break
        except Exception as e:
            print(f"Error trying to read file {path}: {e}")
            try:
                df = pd.read_csv(path, encoding="latin1")
                print(f"Successfully read file with latin1 encoding: {path}")
                break
            except Exception as e2:
                print(f"Error trying to read file {path} with latin1 encoding: {e2}")

if df is None:
    print("File not found or could not be read. Please check the file path.")
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir("."))
    exit()

print("Performing data preprocessing...")
if df.columns[0] == 0:
    df = df.rename(columns={0: 'text'})
elif '0' in df.columns:
    df = df.rename(columns={'0': 'text'})

def extract_title(text, max_length=50):
    if isinstance(text, str):
        first_line = text.split('\n')[0]
        title = first_line[:max_length]
        return title
    return "No title"

df['title'] = df['text'].apply(extract_title)

print("Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    min_df=2,
    max_df=0.9
)
tfidf_matrix = vectorizer.fit_transform(df['text'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

print("Calculating cosine similarity between documents...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_similar_documents(doc_id, df, cosine_sim, top_n=5):
    sim_scores = list(enumerate(cosine_sim[doc_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    doc_indices = [i[0] for i in sim_scores]
    similarity_values = [i[1] for i in sim_scores]
    
    result = df.iloc[doc_indices].copy()
    result['similarity'] = similarity_values
    return result[['title', 'similarity']]

print("\n=== Document Similarity Analysis Example ===")
for i in range(min(5, len(df))):
    print(f"\nDocument {i+1} Title: {df['title'].iloc[i]}")
    similar_docs = get_similar_documents(i, df, cosine_sim)
    print("Most similar documents:")
    print(similar_docs)

print("\nPerforming dimensionality reduction for visualization...")
svd = TruncatedSVD(n_components=2)
reduced_features = svd.fit_transform(tfidf_matrix)

print("Performing K-means clustering...")
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
df['cluster'] = clusters

plt.figure(figsize=(12, 8))
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=df['cluster'], palette='viridis')
plt.title('Semantic Clustering of Machine Learning Papers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('ml_papers_clusters.png')
print("Cluster visualization saved as 'ml_papers_clusters.png'")

print("\n=== Keywords for Each Cluster ===")
def get_cluster_keywords(cluster_id, tfidf_matrix, vectorizer, df, top_n=10):
    cluster_docs_indices = df[df['cluster'] == cluster_id].index
    cluster_tfidf = tfidf_matrix[cluster_docs_indices].toarray()
    
    avg_tfidf = np.mean(cluster_tfidf, axis=0)
    
    feature_names = vectorizer.get_feature_names_out()
    
    keywords = [(feature_names[i], avg_tfidf[i]) for i in range(len(feature_names))]
    keywords.sort(key=lambda x: x[1], reverse=True)
    
    return keywords[:top_n]

for cluster_id in range(num_clusters):
    cluster_size = len(df[df['cluster'] == cluster_id])
    print(f"\nCluster {cluster_id} (contains {cluster_size} documents)")
    keywords = get_cluster_keywords(cluster_id, tfidf_matrix, vectorizer, df)
    print(f"Main keywords: {', '.join([kw[0] for kw in keywords])}")

print("\n=== Cluster Internal Similarity Analysis ===")
cluster_internal_sim = {}
for cluster_id in range(num_clusters):
    cluster_docs = df[df['cluster'] == cluster_id].index.tolist()
    if len(cluster_docs) < 2:
        cluster_internal_sim[cluster_id] = 0
        continue
        
    similarities = []
    for i in range(len(cluster_docs)):
        for j in range(i+1, len(cluster_docs)):
            similarities.append(cosine_sim[cluster_docs[i], cluster_docs[j]])
    
    cluster_internal_sim[cluster_id] = np.mean(similarities)
    print(f"Cluster {cluster_id} internal average similarity: {cluster_internal_sim[cluster_id]:.4f}")

print("\nGenerating similarity heatmap...")
n_docs = min(20, len(df))
sim_subset = cosine_sim[:n_docs, :n_docs]

plt.figure(figsize=(12, 10))
sns.heatmap(sim_subset, annot=False, cmap='YlGnBu')
plt.title('Document Similarity Heatmap (First 20 Documents)')
plt.savefig('similarity_heatmap.png')
print("Similarity heatmap saved as 'similarity_heatmap.png'")

print("\n=== Analysis Complete ===")
print("1. Document similarity analysis: Found the most similar documents for each document")
print("2. Clustering analysis: Grouped documents into", num_clusters, "topic clusters")
print("3. Keyword analysis: Identified main keywords for each cluster")
print("4. Visualization: Generated cluster scatter plot and similarity heatmap") 