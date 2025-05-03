#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:36:16 2025

@author: elenafj
"""

'''
Stylometric Analysis #6

Sentiment Analysis: 
    Examining the emotional tone of the text (positive, negative, neutral) 
    and how it changes throughout the work. This can give insights into 
    the underlying mood and intent behind the language used.
'''

'''
Specifically: It would be interesting to see whether the legal texts refer 
to AI with fear, apprehension, concern, whereas we would expect the 
academic texts to be neutral, appraising, critical, excited by possibilities &
innovations.
'''

'''what are techniques for sentiment analysis which assess sentiment towards a specific concept

ChatGPT said:
Sentiment analysis that focuses on assessing sentiment towards a specific concept or 
aspect (also known as aspect-based sentiment analysis) involves identifying not 
just the overall sentiment of a text but also how a particular entity, feature, 
or concept is perceived. This technique is more granular and detailed, allowing 
for an understanding of sentiments related to specific components of the text, 
rather than just an overall positive or negative sentiment.

Here are some techniques and methods for sentiment analysis that assess sentiment 
towards a specific concept:

1. Aspect-Based Sentiment Analysis (ABSA)
In Aspect-Based Sentiment Analysis (ABSA), the goal is to identify aspects 
(specific features or concepts) mentioned in the text and determine the sentiment 
expressed toward each aspect. For instance, in a product review, aspects might 
include "battery life", "camera quality", and "price."

Steps involved:
Aspect Extraction: Identify aspects or entities mentioned in the text.

Sentiment Classification: Assign sentiment (positive, negative, neutral) to each aspect.

Example:
In a sentence like "The camera quality is great, but the battery life is poor", 
you would identify "camera quality" and "battery life" as aspects and assess 
sentiment separately for each one.

Techniques for Aspect Extraction:
Rule-Based Methods: Use linguistic patterns to extract aspects (e.g., noun phrases, keywords).

Machine Learning Models: Train classifiers (such as SVM, Naive Bayes) to detect 
aspects based on labeled datasets.

Deep Learning Models: Use models like LSTMs or BERT to automatically extract 
aspects from text.

2. Dependency Parsing for Aspect Extraction
Dependency parsing can help identify relationships between words in a sentence, 
which can be useful for identifying aspects and their associated sentiments. 
In a review like "The battery life is terrible, but the camera is great", 
dependency parsing could help identify that "battery life" and "camera" 
are the main aspects, and associate "terrible" and "great" with these aspects.

Approach: Use dependency parsing tools (such as spaCy) to extract aspect-related 
entities and their dependencies.

Benefit: This method can handle complex sentence structures, and it allows 
you to accurately identify aspects even in grammatically complex sentences.

3. Fine-Grained Sentiment Analysis with Pre-trained Models
Some advanced NLP models, such as BERT and its variants (e.g., RoBERTa, DistilBERT), 
can be fine-tuned for specific tasks like aspect-based sentiment analysis. 
This requires training on a labeled dataset where the aspects and sentiments 
are already annotated.

Approach: Fine-tune a pre-trained language model (like BERT) on a dataset with 
labeled aspects and corresponding sentiments.

Benefit: This approach is more accurate because it leverages the model’s 
deep understanding of language nuances and context.

Example:
Using Hugging Face's transformers library, you can fine-tune a pre-trained 
model on an ABSA dataset to automatically classify sentiments toward specific aspects.

4. Lexicon-Based Sentiment Analysis
Lexicon-based approaches use predefined lists of words (or lexicons) with 
associated sentiment scores. These approaches can be adapted to assess sentiment 
toward specific aspects by creating domain-specific lexicons for each aspect.

Approach: Construct sentiment lexicons for different aspects (e.g., "battery life," 
                                                              "performance," "screen quality") 
and match terms in the text to these lexicons to determine sentiment.

Benefit: It’s an interpretable approach and can be efficient for domains where 
aspects are clearly defined and limited.

Example:
In product reviews, lexicons can be created for different features such as "battery," 
"camera," etc., with words like "long-lasting" for battery, or "high quality" for camera quality.

5. Aspect-Category Pairing and Sentiment Classification
Some approaches combine aspect extraction and sentiment classification into a 
single pipeline. These models are designed to both extract aspects and predict 
their corresponding sentiment, allowing for a comprehensive analysis.

Approach: Use sequence labeling or sequence-to-sequence models to extract 
aspect-sentiment pairs.

Benefit: This approach provides fine-grained analysis, capturing both what 
aspects are being discussed and how they are perceived.

6. Opinion Mining using NER (Named Entity Recognition)
Named Entity Recognition (NER) can help identify entities that are important 
in a given text. In the context of sentiment analysis, entities could be considered 
"aspects" (e.g., brands, products, features). This technique is widely used in 
reviewing products or services.

Approach: Use NER tools (like spaCy) to extract entities from the text and 
assess sentiment based on the relationships between the entity and its 
associated sentiment words.

Benefit: Effective for domains like reviews where entities are well defined.

7. Multi-Aspect Sentiment Analysis (MASA)
In MASA, multiple aspects are considered in a single text, and sentiment is 
classified for each aspect. This is useful when analyzing complex texts where 
multiple features are discussed.

Approach: Use neural network models like LSTMs, attention mechanisms, or 
Transformers to analyze the sentiment for each aspect independently while 
considering the overall context.

Benefit: It allows for precise sentiment evaluation of multiple aspects in a single document.

8. Hierarchical Attention Networks (HAN)
In more advanced settings, you can use hierarchical models, like Hierarchical 
Attention Networks (HAN), which treat the document at different granular levels
 (words, sentences) to improve aspect-based sentiment analysis.

Approach: The model focuses on specific parts of the text that are more relevant
 to each aspect and assigns sentiment accordingly.

Benefit: This method provides more accurate results for documents where multiple 
aspects are mentioned in a single sentence or paragraph.

9. Aspect-Specific Word Embeddings
Word embeddings, such as Word2Vec or GloVe, can be fine-tuned or adapted for 
specific aspects of a domain to improve the understanding of sentiments toward specific concepts.

Approach: Train word embeddings specifically on texts related to the concept 
(e.g., tech reviews) so that they can better represent domain-specific terms 
and their associated sentiments.

Benefit: Custom embeddings can capture domain-specific nuances better than 
generic ones.

Tools for Aspect-Based Sentiment Analysis:
Hugging Face Transformers: Pre-trained models for sentiment classification, 
including aspect-based models.

spaCy: For dependency parsing and entity recognition.

VADER: Although it’s designed for general sentiment, you can tailor it for 
aspect-based sentiment analysis by defining custom rules.

Example:
Consider the following product review:

"The camera quality of this phone is excellent, but the battery life is terrible."

Aspect Extraction:

Camera quality

Battery life

Sentiment Classification:

Camera quality: Positive

Battery life: Negative

By using ABSA or other techniques like dependency parsing, you can extract 
the aspects and classify the sentiment towards each one independently
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



