#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 09:02:27 2025

@author: karinacastolo
"""

# Required Libraries
import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Download necessary NLTK data (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Define the function
def analyze_corpora(df1, df2, text_column, ngram_range=(1, 2), top_n_list=[50, 100, 200]):
    """
    This function compares two sets of texts:
    - It shows how many times words or phrases appear in each.
    - It also shows how important those words or phrases are (TF-IDF).
    
    Parameters:
    - df1, df2: your two datasets (Pandas DataFrames)
    - text_column: name of the column that contains the text
    - ngram_range: number of words in a phrase (e.g., (1, 2) = single words and 2-word phrases)
    - top_n_list: how many top phrases you want to extract (top 50, top 100, etc.)
    
    Returns:
    - Two results:
        1. Frequency counts of phrases
        2. TF-IDF scores of phrases
    """

    # List of common phrases that are not meaningful for analysis
    generic_phrases = set([
        "in order", "based on", "according to", "as well", "due to", 
        "can be", "there is", "such as", "in the", "for example", 
        "on the", "as shown", "with respect", "it is", "in this paper", 
        "we present", "we propose", "our results"
    ])

    # ---------------------------
    # Step 1: Count Word Frequencies
    # ---------------------------
    def clean_and_count_ngrams(df):
        # Join all rows of text into one big string
        all_text = " ".join(df[text_column].dropna().astype(str).values)
        all_text = all_text.lower()  # Make all lowercase
        all_text = re.sub(r'[^\w\s]', '', all_text)  # Remove punctuation

        # Tokenize = split text into words
        words = word_tokenize(all_text)

        # Remove common English stopwords like "the", "and", "is"
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in words if w.isalpha() and w not in stop_words]

        # Create list to store all phrases
        all_phrases = []

        # Create unigrams, bigrams, etc.
        for n in range(ngram_range[0], ngram_range[1] + 1):
            ngram_list = list(ngrams(filtered_words, n))
            for group in ngram_list:
                phrase = " ".join(group)
                if phrase not in generic_phrases:  # Remove generic phrases
                    all_phrases.append(phrase)

        # Count each phrase
        return Counter(all_phrases)

    # Apply function to each dataset
    count1 = clean_and_count_ngrams(df1)
    count2 = clean_and_count_ngrams(df2)

    # Get all unique phrases from both corpuses
    all_phrases = set(count1.keys()) | set(count2.keys())

    # Build frequency table
    freq_data = {
        "Phrase": list(all_phrases),
        "Corpus 1 Count": [count1.get(p, 0) for p in all_phrases],
        "Corpus 2 Count": [count2.get(p, 0) for p in all_phrases]
    }

    freq_df = pd.DataFrame(freq_data)
    freq_df["Total Count"] = freq_df["Corpus 1 Count"] + freq_df["Corpus 2 Count"]
    freq_df = freq_df.sort_values(by="Total Count", ascending=False).reset_index(drop=True)

    # Store top-N results in a dictionary
    frequency_results = {}
    for n in top_n_list:
        frequency_results[f"top_{n}"] = freq_df.head(n)
    frequency_results["full"] = freq_df.drop(columns="Total Count")

    # ---------------------------
    # Step 2: TF-IDF Analysis
    # ---------------------------

    # Combine both corpuses into one list
    text_list1 = df1[text_column].dropna().astype(str).tolist()
    text_list2 = df2[text_column].dropna().astype(str).tolist()
    combined_text = text_list1 + text_list2

    # Create labels to know which text belongs to which corpus
    labels = ["Corpus 1"] * len(text_list1) + ["Corpus 2"] * len(text_list2)

    # Use TF-IDF Vectorizer from sklearn
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]{2,}\b',
        min_df=2  # Ignore phrases that appear only once
    )

    # Learn vocabulary and transform the data
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    feature_names = vectorizer.get_feature_names_out()

    # Convert to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_df["label"] = labels

    # Remove generic phrases from TF-IDF columns
    tfidf_df = tfidf_df[[col for col in tfidf_df.columns if col not in generic_phrases] + ["label"]]

    # Calculate average TF-IDF score per phrase for each corpus
    avg_tfidf = tfidf_df.groupby("label").mean().T  # .T = transpose (phrases as rows)
    avg_tfidf["Difference"] = avg_tfidf["Corpus 1"] - avg_tfidf["Corpus 2"]
    avg_tfidf = avg_tfidf.sort_values(by="Difference", key=abs, ascending=False)

    # Return both results
    return frequency_results, avg_tfidf
