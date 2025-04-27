#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:07:07 2025

@author: elenafj
"""

'''
Stylometric Analysis #1

Use Kilgariff's Chi-Squared test as a "goodness of fit" test, 
where the ML texts are the underlying "gold standard" distribution
for the degree of "technicality".
'''

'''
What this method answers:
    Relative word frequency--"do the most frequent words appear as
    often, more often, or less often than expected IF corpus #1 had
    been drawn from the comparison distribution."
    
What this method does NOT answer/analyze/address:
    HOW those words are used (meaning, placement within a sentence, whether they are being "used" "CORRECTLY")
    Bigrams, trigrams
    Phrases
    
Key variations:
    1. Choice of comparison & test distributions -- specifically: 
        a) all ML texts vs. subdividing published & 2024 blurbs
        b) 
    2. Choice of top words -- remove stop words, numbers, etc.?
        a) Y
        b) N
    3. Could expand this method to bigrams & trigrams relatively easily
        (note that the removal of stopwords, etc. would then become
         trickier; analyzing bigrams/trigrams without stopwords would 
         MEAN something very different, i.e. would be a measure of word 
         proximity/clusters rather than specific phrases)
'''

'''
Outputs/results (INCLUDING intermediate/process outputs, to understand 
is working) & expected visualizations:
    - Gridded results for the above setting combinations ('Key variations')
    - Comparative Chi-sq values (no p-values: we KNOW that the null is not true, 
                                 they just provide us with a RELATIVE "gof" measure)
    - Note the effects of removing stopwords
    - Note the effects of looking at bigrams, trigrams
        - And any interactions with removing stopwords
    - Comparison between "technical" wording of different government bodies (EU vs. US vs. UN)
    - List of the top n=500 words (note that choice of n is relatively arbitrary)
        - with & without stopwords
        - bigrams & trigrams
   [- Give citations for methodology]
'''

# Write a function below which will accomplish all of this, taking in the desired settings
# Include a parameter which can allow for "intermediate" outputs, i.e. to examine function internals
    # e.g. "if internals = True: {print X}"




