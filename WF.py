#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 09:02:27 2025

@author: karinacastolo
"""

# # Required Libraries
# import pandas as pd
# import re
# import nltk
# from collections import Counter
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.util import ngrams
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Download required NLTK data (only once)
# nltk.download('punkt')
# nltk.download('stopwords')

# # Define the function
# def analyze_corpora(df1, df2, text_column, ngram_range=(1, 2), top_n_list=[50, 100, 200]):
#     """
#     This function compares two sets of texts:
#     - It shows how many times words or phrases appear in each.
#     - It also shows how important those words or phrases are (TF-IDF).
    
#     Parameters:
#     - df1, df2: your two datasets (Pandas DataFrames)
#     - text_column: name of the column that contains the text
#     - ngram_range: number of words in a phrase (e.g., (1, 2) = single words and 2-word phrases)
#     - top_n_list: how many top phrases you want to extract (top 50, top 100, etc.)
    
#     Returns:
#     - Two results:
#         1. Frequency counts of phrases
#         2. TF-IDF scores of phrases
#     """

#     # List of common phrases that are not meaningful for analysis
#     generic_phrases = set([
#         "in order", "based on", "according to", "as well", "due to", 
#         "can be", "there is", "such as", "in the", "for example", 
#         "on the", "as shown", "with respect", "it is", "in this paper", 
#         "we present", "we propose", "our results"
#     ])

#     # ---------------------------
#     # Step 1: Count Word Frequencies
#     # ---------------------------
#     def clean_and_count_ngrams(df):
#         # Join all rows of text into one big string
#         all_text = " ".join(df[text_column].dropna().astype(str).values)
#         all_text = all_text.lower()  # Make all lowercase
#         all_text = re.sub(r'[^\w\s]', '', all_text)  # Remove punctuation

#         # Tokenize = split text into words
#         words = word_tokenize(all_text)

#         # Remove common English stopwords like "the", "and", "is"
#         stop_words = set(stopwords.words('english'))
#         filtered_words = [w for w in words if w.isalpha() and w not in stop_words]

#         # Create list to store all phrases
#         all_phrases = []

#         # Create unigrams, bigrams, etc.
#         for n in range(ngram_range[0], ngram_range[1] + 1):
#             ngram_list = list(ngrams(filtered_words, n))
#             for group in ngram_list:
#                 phrase = " ".join(group)
#                 if phrase not in generic_phrases:  # Remove generic phrases
#                     all_phrases.append(phrase)

#         # Count each phrase
#         return Counter(all_phrases)

#     # Apply function to each dataset
#     count1 = clean_and_count_ngrams(df1)
#     count2 = clean_and_count_ngrams(df2)

#     # Get all unique phrases from both corpuses
#     all_phrases = set(count1.keys()) | set(count2.keys())

#     # Build frequency table
#     freq_data = {
#         "Phrase": list(all_phrases),
#         "Corpus 1 Count": [count1.get(p, 0) for p in all_phrases],
#         "Corpus 2 Count": [count2.get(p, 0) for p in all_phrases]
#     }

#     freq_df = pd.DataFrame(freq_data)
#     freq_df["Total Count"] = freq_df["Corpus 1 Count"] + freq_df["Corpus 2 Count"]
#     freq_df = freq_df.sort_values(by="Total Count", ascending=False).reset_index(drop=True)

#     # Store top-N results in a dictionary
#     frequency_results = {}
#     for n in top_n_list:
#         frequency_results[f"top_{n}"] = freq_df.head(n)
#     frequency_results["full"] = freq_df.drop(columns="Total Count")

#     # ---------------------------
#     # Step 2: TF-IDF Analysis
#     # ---------------------------

#     # Combine both corpuses into one list
#     text_list1 = df1[text_column].dropna().astype(str).tolist()
#     text_list2 = df2[text_column].dropna().astype(str).tolist()
#     combined_text = text_list1 + text_list2

#     # Create labels to know which text belongs to which corpus
#     labels = ["Corpus 1"] * len(text_list1) + ["Corpus 2"] * len(text_list2)

#     # Use TF-IDF Vectorizer from sklearn
#     vectorizer = TfidfVectorizer(
#         ngram_range=ngram_range,
#         stop_words='english',
#         lowercase=True,
#         token_pattern=r'\b[a-zA-Z]{2,}\b',
#         min_df=2  # Ignore phrases that appear only once
#     )

#     # Learn vocabulary and transform the data
#     tfidf_matrix = vectorizer.fit_transform(combined_text)
#     feature_names = vectorizer.get_feature_names_out()

#     # Convert to DataFrame
#     tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
#     tfidf_df["label"] = labels

#     # Remove generic phrases from TF-IDF columns
#     tfidf_df = tfidf_df[[col for col in tfidf_df.columns if col not in generic_phrases] + ["label"]]

#     # Calculate average TF-IDF score per phrase for each corpus
#     avg_tfidf = tfidf_df.groupby("label").mean().T  # .T = transpose (phrases as rows)
#     avg_tfidf["Difference"] = avg_tfidf["Corpus 1"] - avg_tfidf["Corpus 2"]
#     avg_tfidf = avg_tfidf.sort_values(by="Difference", key=abs, ascending=False)

#     # Return both results
#     return frequency_results, avg_tfidf

# open csv 

import pandas as pd
df = pd.read_csv("/Users/karinacastolo/Documents/Columbia/NLP/nlp_qmss_groupproject/ml_corpuses.csv")
df.rename(columns={'0': 'ml_text'}, inplace=True)

def analyze_corpora(df, text_column, ngram_range=(1, 2), top_n_list=[50], max_features=5000):
    import re
    import pandas as pd
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Basic stopword list
    stop_words = set([
        'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'that', 'this', 'with', 'as', 
        'are', 'it', 'by', 'an', 'be', 'we', 'can', 'or', 'from', 'at', 'not', 'has', 'have'
    ])

    # Generic phrases to exclude
    generic_phrases = set([
        "in order", "based on", "according to", "as well", "due to", 
        "can be", "there is", "such as", "in the", "for example", 
        "on the", "as shown", "with respect", "it is", "in this paper", 
        "we present", "we propose", "our results"
    ])

    # Clean text and count n-grams
    def clean_and_count_ngrams(df):
        all_text = " ".join(df[text_column].dropna().astype(str).values).lower()
        all_text = re.sub(r'[^\w\s]', '', all_text)
        words = [w for w in all_text.split() if w.isalpha() and w not in stop_words]
        all_phrases = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngram_list = zip(*[words[i:] for i in range(n)])
            for group in ngram_list:
                phrase = " ".join(group)
                if phrase not in generic_phrases:
                    all_phrases.append(phrase)
        return Counter(all_phrases)

    # Split into two corpora
    half = len(df) // 2
    df1 = df.iloc[:half].copy()
    df2 = df.iloc[half:].copy()

    # Frequency counts
    count1 = clean_and_count_ngrams(df1)
    count2 = clean_and_count_ngrams(df2)

    all_phrases = set(count1.keys()) | set(count2.keys())
    freq_data = {
        "Phrase": list(all_phrases),
        "Corpus 1 Count": [count1.get(p, 0) for p in all_phrases],
        "Corpus 2 Count": [count2.get(p, 0) for p in all_phrases]
    }
    freq_df = pd.DataFrame(freq_data)
    freq_df["Total Count"] = freq_df["Corpus 1 Count"] + freq_df["Corpus 2 Count"]
    freq_df = freq_df.sort_values(by="Total Count", ascending=False).reset_index(drop=True)

    frequency_results = {f"top_{n}": freq_df.head(n) for n in top_n_list}
    frequency_results["full"] = freq_df.drop(columns="Total Count")

    # TF-IDF Analysis
    text_list1 = df1[text_column].dropna().astype(str).tolist()
    text_list2 = df2[text_column].dropna().astype(str).tolist()
    combined_text = text_list1 + text_list2
    labels = ["Corpus 1"] * len(text_list1) + ["Corpus 2"] * len(text_list2)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b',
        min_df=2,
        max_features=max_features
    )
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    feature_names = vectorizer.get_feature_names_out()

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_df["label"] = pd.Series(labels, index=tfidf_df.index)

    # ✅ Safe filter: keep 'label' intact while removing generic phrases
    cols = [col for col in tfidf_df.columns if col not in generic_phrases and col != "label"]
    tfidf_df = tfidf_df[cols + ["label"]]

    # Group and calculate TF-IDF differences
    avg_tfidf = tfidf_df.groupby("label").mean().T
    avg_tfidf["Difference"] = avg_tfidf["Corpus 1"] - avg_tfidf["Corpus 2"]
    avg_tfidf = avg_tfidf.sort_values(by="Difference", key=abs, ascending=False)

    return frequency_results, avg_tfidf

frequency_results, avg_tfidf = analyze_corpora(
    df=df,
    text_column="ml_text",
    ngram_range=(1, 2),       # (1, 1) for single words only
    top_n_list=[100],         # 👈 This gives you 'top_100' in the result
    max_features=5000
)

print(frequency_results.keys())


# Count total words in the full corpus
all_text = " ".join(df["ml_text"].dropna().astype(str).values).lower()
words = all_text.split()
print("Total number of words in the full corpus:", len(words))

# total number of words without stopwords 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# Use scikit-learn's built-in stopwords
stop_words = ENGLISH_STOP_WORDS

# Combine all text
all_text = " ".join(df["ml_text"].dropna().astype(str).values).lower()

# Remove punctuation
all_text = re.sub(r'[^\w\s]', '', all_text)

# Tokenize (split into words)
words = all_text.split()

# Filter out stopwords
filtered_words = [word for word in words if word not in stop_words]

# Print result
print("Total number of words without scikit-learn stopwords:", len(filtered_words))

#count unique non stop words 

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

# 1. Use built-in English stopwords from scikit-learn
stop_words = ENGLISH_STOP_WORDS

# 2. Combine all text in the 'ml_text' column
all_text = " ".join(df["ml_text"].dropna().astype(str).values).lower()

# 3. Remove punctuation
all_text = re.sub(r'[^\w\s]', '', all_text)

# 4. Tokenize
words = all_text.split()

# 5. Remove stopwords
filtered_words = [word for word in words if word not in stop_words]

# 6. Count unique non-stopwords
unique_non_stopwords = set(filtered_words)
print("Total unique non-stopwords:", len(unique_non_stopwords))

# word frequency non-stopwords 

import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load your CSV (if not already loaded)
# df = pd.read_csv("your_file.csv")  # Uncomment and modify if needed
# df.columns = ['ml_text']           # Make sure column is named correctly

# 1. Combine all text and lowercase
all_text = " ".join(df["ml_text"].dropna().astype(str).values).lower()

# 2. Remove punctuation
all_text = re.sub(r'[^\w\s]', '', all_text)

# 3. Tokenize and remove stopwords
words = all_text.split()
filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]

# 4. Count word frequencies
word_counts = Counter(filtered_words)

# 5. Convert to DataFrame
word_freq_df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"])
word_freq_df = word_freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

# 6. Preview the result
print(word_freq_df.head(10))  # Top 10 words

#count n-grams frequency (non stopword non numeric)

import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Step 1: Combine and clean all text
all_text = " ".join(df["ml_text"].dropna().astype(str).values).lower()
all_text = re.sub(r'[^\w\s]', '', all_text)  # Remove punctuation

# Step 2: Tokenize and filter
words = all_text.split()
filtered_words = [
    word for word in words
    if word not in ENGLISH_STOP_WORDS and not word.isnumeric()
]

# Step 3: N-gram generation function
def generate_ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

# Step 4: Generate bigrams and trigrams
bigrams = generate_ngrams(filtered_words, 2)
trigrams = generate_ngrams(filtered_words, 3)

# Step 5: Count and exclude specific phrases
excluded_phrases = {"et al", "e g", "figure x", "table x"}

ngram_counts = Counter(bigrams + trigrams)
for phrase in excluded_phrases:
    ngram_counts.pop(phrase, None)    

#Dataframe

ngram_df = pd.DataFrame(ngram_counts.items(), columns=["N-gram", "Frequency"])
ngram_df = ngram_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

# Step 6: Preview
print(ngram_df.head(10))

