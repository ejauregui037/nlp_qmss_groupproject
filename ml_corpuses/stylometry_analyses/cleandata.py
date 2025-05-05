#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 14:21:28 2025

@author: elenafj
"""

'''
Pre-processing the data: to give to different analytical methods/algorithms
*This script should be sourced by each of the other analytical scripts*
'''

'''
OUTPUT DFs, FOR USE IN OTHER SCRIPTS -- full list:
    
    Label categories:
        "version 1" =  ml_technical & legal --- least useful
        "version 2" = [ml_published, ml_2024] & [EU, UN, US] *** -- USEFUL
        "version 3" = ml_technical & [EU, UN, US] *** -- USEFUL
        "version 4" = [ml_published, ml_2024] & legal *** -- USEFUL
    
    - [version_1, version_2, version_3, version_4]: lowercased
    
    - [version_1n, version_2n, version_3n, version_4n]: lowercased, no numbers
    - then, for each: a list version_1n123 x 4 of len=3, where the first item has unigrams, then bigrams, then trigrams
        
    - [version_1s, version_2s, version_3s, version_4s]: lowercased, no numbers, no stopwords
    - then, for each: a list version_1s123 x 4 of len=3, where the first item has unigrams, then bigrams, then trigrams
        # note: these are "pseudo-ngrams": what I am curious about here is how keywords of interest are used in proximity to each other! i.e.
        # is it common for technical papers to use the word "optimize" before "model", whereas in legal documents it might be more common to use the term "optimize" before "safety"?
            
    - [version_1c, version_2c, version_3c, version_4c]: capitalization maintained            
    - [version_1cn, version_2cn, version_3cn, version_4cn]: capitalization maintained, no numbers
    - [version_1ct, version_2ct, version_3ct, version_4ct]: capitalization maintained; each corpus chopped into a list of sentence strings
        # Keeping capitalization has the benefit of allowing us to distinguish sentences. 
        # Therefore, no reason to have a stopword-free version here            
        
'''
            
'''
IN SCRIPTS WHICH SOURCE THIS ONE:
    source(cleandata.py)
THEN: (delete the rest/only keep the relevant ones in other scripts)
    # List of variables you want to keep
    keep_vars = {'important_var1', 'important_var2'}
    
    # Delete everything in globals() except for the ones you want to keep and system variables
    for var in list(globals()):
        if var not in keep_vars and not var.startswith("__"):
            del globals()[var]
'''

# =============================================================================
# Load & prep data
# =============================================================================

import pandas as pd
import nltk

# technical corpuses (ML)
mlpapersummaries = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_papersummaries.csv")
mlcorps = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_corpuses.csv")
mlcorps_noref = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_corpuses_noref.csv")

# legal data
legal_dat = pd.read_pickle("~/Downloads/cleaned_df.pkl") # trying out Haley's data -- change location

# =============================================================================
# Prep # 1: combine all of these texts into a single df, with "authors"
# =============================================================================

# less specific: "author"
mlcorps_noref['author'] = "ml_technical"
mlpapersummaries['author'] = "ml_technical"
legal_dat['author'] = "legal"

# more specific: "author1"
mlcorps_noref['author1'] = "ml_published"
mlpapersummaries['author1'] = "ml_2024"
legal_dat['author1'] = legal_dat['region_label']

# standardize column names for extracted text
legal_dat = legal_dat.rename(columns={'Extracted Text': 'corpuses'})
mlcorps_noref = mlcorps_noref.rename(columns={'0': 'corpuses'})
mlpapersummaries = mlpapersummaries.rename(columns={'0': 'corpuses'})

# Version 1: all ML together;  all legal together (same "author")
version_1 = pd.concat([legal_dat[['corpuses','author']],
                       mlcorps_noref[['corpuses','author']], 
                       mlpapersummaries[['corpuses','author']] ],
                       axis=0, ignore_index=True) # reset indices
# Version 2: ML differs by published vs. more casual style & updated methods;  legal documents separated by governing body
version_2 = pd.concat([legal_dat[['corpuses','author1']],
                       mlcorps_noref[['corpuses','author1']], 
                       mlpapersummaries[['corpuses','author1']] ],
                       axis=0, ignore_index=True)
# Version 3: all ML together;  (!!!) legal documents separated by governing body
to_delete = legal_dat.copy()
to_delete['author'] = to_delete['author1'] # to enable pd.concat; must have same colname as other dfs
version_3 = pd.concat([to_delete[['corpuses','author']],
                       mlcorps_noref[['corpuses','author']], 
                       mlpapersummaries[['corpuses','author']] ],
                       axis=0, ignore_index=True)

# Version 3: (!!!) ML differs by published vs. more casual style & updated methods;  all legal together (same "author")
to_delete = legal_dat.copy()
to_delete['author1'] = to_delete['author'] # to enable pd.concat; must have same colname as other dfs
version_4 = pd.concat([to_delete[['corpuses','author1']],
                       mlcorps_noref[['corpuses','author1']], 
                       mlpapersummaries[['corpuses','author1']] ],
                       axis=0, ignore_index=True)

# rename columns for consistency
version_2 = version_2.rename(columns={'author1': 'author'})
version_4 = version_4.rename(columns={'author1': 'author'})

# for all: replace any number of spaces >1 with one space only
import re
def reg_extraspace(string_in):
    cleaned_text = re.sub(r'\s{2,}', ' ', string_in) # replace 2 or more spaces with 1 space
    return cleaned_text
def remove_extraspaces(df_in, colname):
    df_in[colname] = df_in[colname].apply(reg_extraspace) # apply to the appropriate column of corpuses
    return df_in
# apply functions to remove extra spaces
version_1 = remove_extraspaces(version_1, 'corpuses')
version_2 = remove_extraspaces(version_2, 'corpuses')
version_3 = remove_extraspaces(version_3, 'corpuses')
version_4 = remove_extraspaces(version_4, 'corpuses')

# =============================================================================
# Lowercase & non-lowercase versions
# =============================================================================

# Capitalized versions
version_1c = version_1.copy()
version_2c = version_2.copy()
version_3c = version_3.copy()
version_4c = version_4.copy()

# Lowercase the tokens so that the same word, capitalized or not, counts as one word
version_1['corpuses'] = [corpus.lower() for corpus in version_1['corpuses']]
version_2['corpuses'] = [corpus.lower() for corpus in version_2['corpuses']]
version_3['corpuses'] = [corpus.lower() for corpus in version_3['corpuses']]
version_4['corpuses'] = [corpus.lower() for corpus in version_4['corpuses']]

# =============================================================================
# Prep # 2: create versions without numbers (lowercased)
# =============================================================================

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))

# functions to remove numbers & stopwords
def clean_numeric(string_in):
    no_numstring = re.sub(r'\d+', '', string_in) # remove all numbers from the string
    return no_numstring
def clean_stopwords(string_in):
    word_list = [word for word in word_tokenize(string_in) if word.lower() not in stop_words] # remove stopwords
    return " ".join(word_list)

# automate the cleaning process for each version df
def update_version(version_df, fxn_in):
    out_1 = version_df.copy()
    out_1['corpuses'] = out_1['corpuses'].apply(fxn_in)
    return out_1

# run for each corpus in each df with "apply" -- remove numbers
version_1n = update_version(version_1, clean_numeric)
version_2n = update_version(version_2, clean_numeric)
version_3n = update_version(version_3, clean_numeric)
version_4n = update_version(version_4, clean_numeric)
# extra spaces have appeared again
version_1n = remove_extraspaces(version_1n, 'corpuses')
version_2n = remove_extraspaces(version_2n, 'corpuses')
version_3n = remove_extraspaces(version_3n, 'corpuses')
version_4n = remove_extraspaces(version_4n, 'corpuses')

# =============================================================================
# Prep # 3: create versions without numbers & stopwords (lowercased)
# =============================================================================
# run for each corpus in each df with "apply"
version_1s = update_version(version_1n, clean_stopwords)
version_2s = update_version(version_2n, clean_stopwords)
version_3s = update_version(version_3n, clean_stopwords)
version_4s = update_version(version_4n, clean_stopwords)

# =============================================================================
# Prep # 4: ngram versions (lowercase, numbers excluded but not stopwords) BUT ALSO: no punctuation/special characters!! VERY IMPORTANT FOR THE NGRAMS
# =============================================================================
from nltk import ngrams, word_tokenize

# create a list of ngrams -- 1,2,3
def ngrams_in(string_in,ngrams_in):
    tokens = word_tokenize(string_in)
    # Remove punctuation from tokens
    import string
    tokens = [token for token in tokens if token not in string.punctuation]
    # now generate the ngrams
    grams_n = list(ngrams(tokens,ngrams_in)) # ngrams_in = 1 for unigram, 2 for bigram, etc...
    return grams_n
def ngrams_custom(df_in, n_in):
    df_ngram = df_in.copy()
    df_ngram['ngram1'] = df_ngram['corpuses'].apply(lambda x: ngrams_in(x, n_in))
    return df_ngram

version_1n123 = [ngrams_custom(version_1n, 1), 
                 ngrams_custom(version_1n, 2), 
                 ngrams_custom(version_1n, 3)]
version_2n123 = [ngrams_custom(version_2n, 1), 
                 ngrams_custom(version_2n, 2), 
                 ngrams_custom(version_2n, 3)]
version_3n123 = [ngrams_custom(version_3n, 1), 
                 ngrams_custom(version_3n, 2), 
                 ngrams_custom(version_3n, 3)]
version_4n123 = [ngrams_custom(version_4n, 1), 
                 ngrams_custom(version_4n, 2), 
                 ngrams_custom(version_4n, 3)]

# =============================================================================
# Prep # 5: ngram versions (lowercase, numbers AND stopwords removed) BUT ALSO: no punctuation/special characters!! VERY IMPORTANT FOR THE NGRAMS
# =============================================================================

version_1s123 = [ngrams_custom(version_1s, 1), 
                 ngrams_custom(version_1s, 2), 
                 ngrams_custom(version_1s, 3)]
version_2s123 = [ngrams_custom(version_2s, 1), 
                 ngrams_custom(version_2s, 2), 
                 ngrams_custom(version_2s, 3)]
version_3s123 = [ngrams_custom(version_3s, 1), 
                 ngrams_custom(version_3s, 2), 
                 ngrams_custom(version_3s, 3)]
version_4s123 = [ngrams_custom(version_4s, 1), 
                 ngrams_custom(version_4s, 2), 
                 ngrams_custom(version_4s, 3)]

# =============================================================================
# Prep # 6: maintain capitalization, numbers removed 
# =============================================================================
version_1cn = update_version(version_1c, clean_numeric)
version_2cn = update_version(version_2c, clean_numeric)
version_3cn = update_version(version_3c, clean_numeric)
version_4cn = update_version(version_4c, clean_numeric)
# extra spaces have appeared again
version_1cn = remove_extraspaces(version_1cn, 'corpuses')
version_2cn = remove_extraspaces(version_2cn, 'corpuses')
version_3cn = remove_extraspaces(version_3cn, 'corpuses')
version_4cn = remove_extraspaces(version_4cn, 'corpuses')

# =============================================================================
# Prep # 7: maintain capitalization, numbers removed - chopped into sentences
# =============================================================================

# split into "sentences" (roughly) by rule:
    # If you see Capital + lowercasex15 (20?) (or space), count that Capital as the beginning of a sentence
    
def split_sentence(string_in): # for legal texts, which do not have periods to indicate sentences...
    # placeholder before each detected 'sentence' start
    pattern = r'(?<=[a-z ])(?=([A-Z][a-z ]{60}))' # 60 chosen by eyeballing; no standard way to do this. Difficult without periods in the text
    parts = re.split(pattern, string_in)
    # Join matched text parts; ignore empty or unmatched groups
    sentences = [''.join(parts[i+1:i+2]).strip() for i in range(1, len(parts), 2)]
    return sentences

def split_proper(string_in): # split sentences where period are present, i.e. in the ML texts
    pattern = r'(?<=\.\s)(?=[A-Z])' # split by periods, but only if they are followed by a space and then a capital letter
    sentences = re.split(pattern, string_in)
    # sentences = string_in.split('.') # there are messy periods: e.g. "et al." -- ignore those...
    #Remove extra spaces and filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# apply differentially to the appropriate rows in our dfs
def sentences_ourdfs(df_in):
    df_sentences = df_in.copy()
    df_sentences1 = df_sentences.iloc[0:18,] # the first 18 are legal, however they are labeled
    df_sentences2 = df_sentences.iloc[18:len(df_sentences),]
    # apply appropriate fxns
    df_sentences1 = update_version(df_sentences1, split_sentence)
    df_sentences2 = update_version(df_sentences2, split_proper)
    # combine
    sentences_df = pd.concat([df_sentences1, df_sentences2], axis=0, ignore_index=True)
    return sentences_df

# apply to capitalized text without numbers -- lEGAL TEXTS
# for the ML texts, split by periods!
version_1ct = sentences_ourdfs(version_1cn)
version_2ct = sentences_ourdfs(version_2cn)
version_3ct = sentences_ourdfs(version_3cn)
version_4ct = sentences_ourdfs(version_4cn)

# =============================================================================
# Lists of outputs: clean output when sourcing this script
# =============================================================================

# Clean environment

# List of variables you want to keep
keep_vars = {'version_1', 
             'version_1c', "version_1cn", "version_1ct", 
             "version_1n", "version_1n123", # "123" is a list of dfs
             "version_1s", "version_1s123", # "123" is a list of dfs
              
             'version_2', 
             'version_2c', "version_2cn", "version_2ct", 
             "version_2n", "version_2n123",
             "version_2s", "version_2s123",
             
             'version_3', 
             'version_3c', "version_3cn", "version_3ct", 
             "version_3n", "version_3n123",
             "version_3s", "version_3s123",
             
             'version_4', 
             'version_4c', "version_4cn", "version_4ct", 
             "version_4n", "version_4n123",
             "version_4s", "version_4s123"}

# Delete everything in globals() except for the ones you want to keep and system variables
for var in list(globals()):
    if var not in keep_vars and not var.startswith("__"):
        del globals()[var]



