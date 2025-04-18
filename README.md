Involved analyzing semantic similarities among 170 machine learning-related research articles. The workflow included:

Scripts and Tools Used
NLP Group Project notebooks (.ipynb): Extracted text from PDFs and stored it in structured formats like .pickle.

project_scrapeMLpapers.py: Scraped ML paper metadata from GitHub and saved it to a CSV.

WF.py: Compared two text corpora using word frequency and TF-IDF scoring.

Analysis Workflow
TF-IDF Vectorization:

Used TfidfVectorizer to convert each document into a 5,000-dimensional vector based on most frequent terms.

Cosine Similarity Calculation:

Measured document similarity using cosine similarity; top similar documents for the first 5 papers were listed.

Dimensionality Reduction & Clustering:

Applied TruncatedSVD for dimensionality reduction and KMeans clustering.

Keyword Extraction per Cluster:

Averaged TF-IDF scores within each cluster to identify top 10 keywords.

Internal Similarity Calculation:

Computed average internal similarity within each cluster to assess cohesion.

📊 Results
Five distinct thematic clusters were identified:

Cluster 0: Translation & language modeling

Cluster 1: Deep learning architecture

Cluster 2: Image classification (Inception, layers)

Cluster 3: Generative models & sequence data

Cluster 4: CNNs, object detection, and image features

Internal similarity scores ranged from 0.1181 to 0.1782, indicating variable cohesion across clusters.

A similarity heatmap was generated to visualize document relationships.
