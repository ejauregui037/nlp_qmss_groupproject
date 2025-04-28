#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:37:42 2025

@author: elenafj
""" 

'''
Stylometric Analysis #7

Cohesive Harmony and Referential Chains: 
    This refers to how words, phrases, or concepts reference each other 
    within the text. By identifying how an author shifts between ideas 
    and refers to entities across a text, you can study their 
    conceptualization of events or characters and how they construct meaning.
'''

'''
Specifically: Evaluate cohesive harmony for SEMANTIC CONSISTENCY:
        Assess whether concepts/terms/ideas are consistently presented throughout
        a text/corpus--whether their maintain their meaning without introducing
        contradictions or ambiguities
        
For us: are they consistent in how they refer to...
    AI, ML, etc.
'''

'''
NLP approaches:

ChatGPT said:
Evaluating cohesive harmony for semantic consistency using NLP methods involves 
assessing whether concepts, terms, or ideas are consistently presented throughout 
a text or corpus, ensuring they maintain their meaning without introducing contradictions 
or ambiguities. Here's how you can approach this evaluation systematically using NLP techniques:

1. Lexical Cohesion (Synonymy and Repetition)
Lexical cohesion refers to how words and phrases are connected semantically 
across a text. You can evaluate this aspect using the following methods:

Word Embeddings (e.g., Word2Vec, GloVe, FastText): These models map words to 
dense vector representations based on their semantic similarity. By using word 
embeddings, you can measure the similarity between terms used in different parts 
of the text and check whether different terms used to refer to the same concept 
are consistent in meaning. For example, if "AI" and "artificial intelligence" 
are used in different parts of the text, you can measure how close these terms 
are in the embedding space. If they are close, it suggests lexical cohesion 
is maintained.

Similarity Score: Measure the cosine similarity between the vector 
representations of key terms. If terms that refer to the same concept 
(e.g., "AI," "machine learning," "deep learning") have a high similarity
 score, you can conclude that the text is semantically consistent in its 
 lexical choices.

Semantic Role Labeling (SRL): SRL helps identify the relationships between 
words in a sentence, such as the action and its participants. This can help 
determine if the core concepts in a sentence are consistently used with the
 same roles throughout the text. For example, if "climate change" is the 
 subject of a sentence, it should consistently appear in similar syntactic
 roles across the text (e.g., as the subject of statements about global 
                        warming or carbon emissions).

2. Named Entity Recognition (NER) and Entity Disambiguation
Named Entity Recognition (NER) is an essential technique for identifying and 
categorizing key concepts in a text (such as names of people, organizations, 
                                     locations, etc.). Evaluating the 
consistency of these entities is important for ensuring semantic consistency.

NER Models: Using pre-trained NER models (e.g., spaCy, Stanford NLP, 
                                          BERT-based models), you can extract 
entities like "AI," "machine learning," or "climate change" and track how they 
are referred to throughout the corpus. Ensuring that the same entity is always 
recognized and referred to correctly (without mixing it with other similar
                                      terms or introducing contradictions) 
is crucial for semantic consistency.

Entity Disambiguation: Once entities are recognized, they need to be 
disambiguated (i.e., determining which entity a term refers to). 
For example, if "apple" appears in the text, we need to determine whether 
it refers to the fruit or the tech company. Tools like ELMo (Embeddings 
                                                             from Language Models) 
or BERT can help disambiguate these references. Ensuring that the same concept 
is consistently identified and not confused with unrelated entities helps 
maintain semantic consistency.

3. Coreference Resolution
Coreference resolution is a process that determines when different phrases 
(such as pronouns) refer to the same entity or concept. This is critical 
for maintaining semantic consistency, as improper or inconsistent coreference
 can break the flow of the text and create ambiguity.

Coreference Resolution Models: Tools like spaCy, AllenNLP, and CoreNLP can 
help resolve coreferences in a text. These models identify when a pronoun 
(e.g., "he," "it," "they") or a noun phrase (e.g., "the company") refers to the
 same entity mentioned earlier in the text. For example, if the text refers to 
 "AI" and then uses "it," the coreference model will ensure that "it" refers
 back to "AI" consistently. Checking that the correct references are maintained
 throughout the corpus helps ensure semantic consistency.

Consistency of Coreference Chains: Evaluate whether coreference chains are
 maintained without introducing ambiguity. If "AI" is referred to in one part 
 of the text and then suddenly referred to as "the technology" or "the system" 
 without explanation, it might indicate a lack of cohesive harmony in terms of
 consistent reference.

4. Topic Modeling for Semantic Consistency
Topic modeling is an unsupervised technique that can help identify the main 
themes or topics within a text or corpus. Using techniques like Latent Dirichle
t Allocation (LDA) or Non-negative Matrix Factorization (NMF), you can identify 
clusters of words that co-occur and are likely related to the same semantic concept.

Topic Consistency: After performing topic modeling, you can analyze how 
consistent the topics are across the text. For example, if a text about 
"AI" starts discussing its applications in healthcare, it should maintain
 this theme consistently across different sections or documents in the corpus.
 If topics suddenly shift to unrelated concepts (e.g., discussing "AI" in the
                                                 context of climate change without
                                                 clear justification), it could 
 indicate a breakdown in cohesive harmony and semantic consistency.

5. Sentence-Level Semantic Consistency
Sometimes, ensuring semantic consistency also means checking that the meaning 
of sentences remains logical and consistent throughout the corpus.

Semantic Textual Similarity (STS): You can use models like BERT, RoBERTa, 
or Sentence-BERT to compute sentence similarity scores. By comparing the 
meaning of sentences or paragraphs within the text, you can evaluate if 
they consistently express the same concepts or ideas. For example, if a 
sentence in one part of the text says, "AI is used in medicine," and a 
later sentence says, "The technology in healthcare includes AI," you can
 measure whether these sentences are semantically consistent in their 
 portrayal of the concept.

Textual Entailment: Natural Language Inference (NLI) models can help determine 
whether one sentence logically entails another, which is helpful for checking 
whether claims made in one part of the text are consistent with claims in 
other parts. For example, if one part of the text asserts, "AI is transforming 
healthcare," and another part says, "AI has no impact on healthcare," these 
two claims would contradict each other, indicating inconsistency.

6. Evaluating Conceptual Shift or Drift
Conceptual shift refers to the abrupt or unclear transition from one concept 
to another, which can break semantic consistency.

Textual Analysis with Clustering: You can use clustering techniques like 
k-means or DBSCAN to group similar sentences or documents together based 
on their semantic content. If a concept suddenly shifts without clear
 signaling (e.g., from "AI" to "automation" or from "machine learning" 
            to "deep learning"), clustering analysis can detect such shifts,
 signaling a potential breakdown in cohesive harmony and semantic consistency.

7. Evaluating Lexical Consistency Using BERT or GPT
Modern transformer-based models like BERT or GPT can be used for fine-grained 
analysis of semantic consistency.

Contextualized Word Embeddings: You can check if a term's meaning stays 
consistent across the text by looking at how its contextualized embeddings 
change. If a concept is used in a different context or with a different 
meaning in one part of the text, the embeddings will reflect this change. 
By examining these shifts, you can evaluate whether the term has been used 
consistently or not.

Summary of NLP Techniques for Evaluating Semantic Consistency:
Word Embeddings (Word2Vec, GloVe, FastText) for measuring lexical cohesion and term consistency.

Named Entity Recognition (NER) and Entity Disambiguation to track consistent 
use of entities or concepts across the text.

Coreference Resolution to ensure consistent use of pronouns and noun phrases
 referring to the same concept.

Topic Modeling (LDA, NMF) to evaluate whether topics remain consistent and 
logically connected across a corpus.

Semantic Textual Similarity (STS) and Textual Entailment models to ensure 
that sentences or claims are semantically aligned.

Clustering and Drift Detection to find conceptual shifts or sudden 
changes in meaning.

Contextualized Embeddings from BERT, GPT, or other transformer models
 to track changes in word meaning and ensure consistent use of concepts.

By leveraging these NLP methods, you can evaluate whether a text or corpus 
maintains semantic consistency and cohesive harmony across different sections,
 ensuring that the concepts remain logically connected and meaningfully 
 presented throughout the discourse.

'''



