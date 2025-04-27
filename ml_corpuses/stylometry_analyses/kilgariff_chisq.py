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
        b) all legal texts vs. subdividing based on geographic/legal entity
    2. Choice of Top words -- remove stop words, numbers, etc.?
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

# =============================================================================
# TO BUILD:
# =============================================================================

# Write a function below which will accomplish all of this, taking in the desired settings
# Include a parameter which can allow for "intermediate" outputs, i.e. to examine function internals
    # e.g. "if internals = True: {print X}"

# =============================================================================
# Load & prep data
# =============================================================================

import pandas as pd

# technical corpuses (ML)
mlpapersummaries = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_papersummaries.csv")
mlcorps = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_corpuses.csv")
mlcorps_noref = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_corpuses_noref.csv")

# legal data
legal_dat = pd.read_pickle("~/Downloads/cleaned_df.pkl") # trying out Haley's data -- change location

# Prep: combine all of these texts into a single df, with "authors"

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
                       mlpapersummaries[['corpuses','author']] ])
# Version 2: ML differs by published vs. more casual style & updated methods;  legal documents separated by governing body
version_2 = pd.concat([legal_dat[['corpuses','author1']],
                       mlcorps_noref[['corpuses','author1']], 
                       mlpapersummaries[['corpuses','author1']] ])
# Version 3: all ML together;  (!!!) legal documents separated by governing body
to_delete = legal_dat.copy()
to_delete['author'] = to_delete['author1'] # to enable pd.concat; must have same colname as other dfs
version_3 = pd.concat([to_delete[['corpuses','author']],
                       mlcorps_noref[['corpuses','author']], 
                       mlpapersummaries[['corpuses','author']] ])

# Version 3: (!!!) ML differs by published vs. more casual style & updated methods;  all legal together (same "author")
to_delete = legal_dat.copy()
to_delete['author1'] = to_delete['author'] # to enable pd.concat; must have same colname as other dfs
version_4 = pd.concat([to_delete[['corpuses','author1']],
                       mlcorps_noref[['corpuses','author1']], 
                       mlpapersummaries[['corpuses','author1']] ])

# rename columns for consistency
version_2 = version_2.rename(columns={'author1': 'author'})
version_4 = version_4.rename(columns={'author1': 'author'})

# Lowercase the tokens so that the same word, capitalized or not, counts as one word
version_1['corpuses'] = [corpus.lower() for corpus in version_1['corpuses']]
version_2['corpuses'] = [corpus.lower() for corpus in version_2['corpuses']]
version_3['corpuses'] = [corpus.lower() for corpus in version_3['corpuses']]
version_4['corpuses'] = [corpus.lower() for corpus in version_4['corpuses']]

# Now: create versions without stopwords
version_1s = 
version_2s = 
version_3s = 
version_4s = 

# =============================================================================
# 1. Kilgariff's Chi-Sq Method
# =============================================================================
'''
https://programminghistorian.org/en/lessons/introduction-to-stylometry-with-python
Excellent source -- 
* Second Stylometric Test: Kilgariff’s Chi-Squared Method
(note: while this is simpler than "Third Stylometric Test: John Burrows’ Delta Method (Advanced)", 
 I think it is actually more applicable to what we are running here! than the delta method)
    - Note: this implies that a dictionary of lists would be a better storage mechanism than a df! That's
    a super easy conversion from list format especially, so maybe I should just pickle the list?
'''

# Approach:
'''
# Let the "authors" be 
# 1. the technical corpuses -- combined (i.e. ML researchers)
# 2. different legal documents -- 1 corpus each
# 3. combinations of those corpuses -- i.e. by regulating body (EU vs. UN)
'''

""" 
Begin stylometry with those author designations.
"""

# Who are the authors we are analyzing?
# I can just cite these in a loop, as the colum in the df

import nltk

# Calculate chisquared for each of the two candidate authors
def kilgariff_chisq(df_corp_auth, ref_list, compar_grp, n_mostcomm, internals):
    '''
    Arguments:
        df_corp_auth: pre-processed df
        ref_list: the list of column names for the "reference groups" (i.e. one or two ML column names). This is the "reference distribution"
        compar_grp: column name for the "comparison groups" (i.e. one or more of the legal designations)
            Note: keep this to one, not a list. If you want to do multiple comparison list, just run this in a loop.
        n_mostcomm: the top "n" most frequent tokens will be used for this comparison
    '''
    
    # Comparison distribution: the combined ml corpuses [as a single mega-corpus]
    refML_dist =  " ".join(df_corp_auth[df_corp_auth['author']==ref_list]['corpuses'].tolist()) 
    
    # Count the tokens for each of the words that can be found in this larger corpus.
    refML_freq_dist =  nltk.FreqDist(refML_dist.split())
    # and the total number of tokens, to get frequency
    refML_totaltokens = sum(refML_freq_dist.values())
    
    if internals == True: 
        print(refML_totaltokens) # INTERNALS
    
    # Select the n most common words in the larger corpus.
    refML_most_common = list(refML_freq_dist.most_common(n_mostcomm)) # n = 500; could be an input to the fxn
    
    if internals == True: 
        print(refML_most_common[1:5]) # INTERNALS
        
    # make a joint corpus for each 'comparison' group of the legal
    compar_dist =  " ".join(df_corp_auth[df_corp_auth['author']==compar_grp]['corpuses'].tolist()) 
    # and the total number of tokens, to get frequency
    compar_freq_dist =  nltk.FreqDist(compar_dist.split())
    compar_totaltokens = sum(compar_freq_dist.values())
    
    if internals == True: 
        print(compar_totaltokens) # INTERNALS
        
    # calculate goodness-of-fit chi-sq for this comparison
    chisquared = 0
    for word, joint_count in refML_most_common:
    
        # How often do we see this word...
        # in ref ML corpus(es)
        refML_count = refML_dist.count(word) 
    
        # Expected & observed counts in the legal corpus of interest
        expected_compar_count = (refML_count/refML_totaltokens) * compar_totaltokens
        observed_compar_count = compar_dist.count(word) # legal docs
        
        # Add the word's contribution to the chi-squared statistic
        chisquared += ((observed_compar_count - expected_compar_count)**2) / expected_compar_count
    
    if internals == True: 
        print("The Chi-squared statistic for candidate", compar_grp, "is", chisquared)
        
    return chisquared
    
# =============================================================================
# Generalized version of the fxn for NGRAMS
# =============================================================================






# =============================================================================
# Apply to the legal corpora -- with & without stopwords
# =============================================================================

# Higher chi-sq means that it probably is NOT from the comparison distribution
# Exact statistical interpretation::
    # # # 
    # 

kilgariff_chisq(df_corp_auth = version_3, 
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

kilgariff_chisq(df_corp_auth = version_3, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

kilgariff_chisq(df_corp_auth = version_3, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

# Now: compare to the different ML versions/regions

# Now: apply to the versions without stopwords

# RESULT: WHICH ARE MORE SIMILAR (RELATIVELY) TO THE TECHNICAL TEXTS--of EU/US/UN--AND WHICH ARE THEY MORE SIMILAR TO (causal/recent vs. published/old)


# =============================================================================
# Applied to ngrams
# =============================================================================


# Now: compare to the different ML versions/regions

# Now: apply to the versions without stopwords

# RESULT: WHICH ARE MORE SIMILAR (RELATIVELY) TO THE TECHNICAL TEXTS--of EU/US/UN--AND WHICH ARE THEY MORE SIMILAR TO (causal/recent vs. published/old)








