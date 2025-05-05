#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:09:05 2025

@author: elenafj
"""

'''
Stylometric Analysis #2

Dependency Parsing:
    Examines the grammatical relationships between words in sentences. 
    This can reveal how authors structure their sentences, which can be
    useful for identifying stylistic traits like sentence complexity, 
    coherence, and preferred syntactic structures.
'''

# =============================================================================
# Load prepped dataframes 
# =============================================================================
# import sys
# sys.path.append("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/stylometry_analyses")
# import cleandata # cleandata.py
exec(open('/Users/elenafj/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/stylometry_analyses/cleandata.py').read())
# takes a minute to run, but loads them all -- more reliable than trying to import repeatedly
# this is R-type coding: source()

# =============================================================================
# Select the dfs relevant to this script
# =============================================================================
# keep_vars = {'version_1', 
#              'version_1c', "version_1cn", "version_1ct", 
#              "version_1n", "version_1n123", # "123" is a list of dfs
#              "version_1s", "version_1s123", # "123" is a list of dfs
              
#              'version_2', 
#              'version_2c', "version_2cn", "version_2ct", 
#              "version_2n", "version_2n123",
#              "version_2s", "version_2s123",
             
#              'version_3', 
#              'version_3c', "version_3cn", "version_3ct", 
#              "version_3n", "version_3n123",
#              "version_3s", "version_3s123",
             
#              'version_4', 
#              'version_4c', "version_4cn", "version_4ct", 
#              "version_4n", "version_4n123",
#              "version_4s", "version_4s123"}

# # Delete everything in globals() except for the ones you want to keep and system variables
# for var in list(globals()):
#     if var not in keep_vars and not var.startswith("__"):
#         del globals()[var]

# =============================================================================
# 
# =============================================================================




# Use spacy to get "Most frequent dependency relations." across each of the corpuses!
    # Note that these are part-of-word dependencies... so I will need to winnow them down to words I am interested in...
    # Also: use spacy built-in pre-processing to get corpora to lists of sentences as strings

import spacy # Note: this won't run in spacy. Using pip install in VSCode (switched IDEs) instead

# Load (pre-trained) spaCy model
spacy_mod = spacy.load("en_core_web_sm")

# Corpus
test = version_1.iloc[1,0]

# Dependency parsing
for doc in spacy_mod.pipe(corpus):
    for token in doc:
        print(f"{token.text}: {token.dep_} -> {token.head.text}")

import spacy
from collections import Counter

# Load spaCy's language model
nlp = spacy.load("en_core_web_sm")

# Example corpus (you can replace this with your actual corpus)
corpus = [
    "SpaCy is an NLP library that makes working with text easier.",
    "This is an example sentence for testing dependency parsing.",
    "Dependency parsing is the task of analyzing the grammatical structure of a sentence."
]

# Counter to store the frequency of dependency relations
dependency_counts = Counter()

# Process each sentence in the corpus
for text in corpus:
    doc = nlp(text)  # Process the sentence with spaCy

    # Count the dependencies for each token in the sentence
    for token in doc:
        dependency_counts[token.dep_] += 1

# Get the most common dependencies
most_common_dependencies = dependency_counts.most_common()

# Print the results
print("Most Frequent Dependency Relations:")
for dep, count in most_common_dependencies:
    print(f"{dep}: {count}")


# ^^^ run this & compare between the various corpora! Interesting descriptive results...
# What I would love to see: dependencies between key technical terms that do NOT match up in legal documents
# vs. ML docs...  

























