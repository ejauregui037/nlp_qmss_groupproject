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
        print(refML_most_common[1:20]) # INTERNALS
        
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
# Apply to the legal corpora -- with & without stopwords
# =============================================================================

# Higher chi-sq means that it probably is NOT from the comparison distribution
# Exact statistical interpretation::
    # # # 
    # 

# =============================================================================
# 'version' ; not cleaned at all
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3,  # bad -- all the top words are stopwords
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================
# Without numbers
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3n,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3n, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3n, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without stopwords and numbers
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3s, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3s, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the two ML corpora
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
EU_chisq1 = kilgariff_chisq(df_corp_auth = version_2s,  
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

print("EU vs. ML published up to 2016: " + str(EU_chisq) + ", EU vs. ML2024: " + str(EU_chisq1))
print("EU vs. ML published up to 2016: " + str(round(EU_chisq)) + ", EU vs. ML2024: " + str(round(EU_chisq1))) # curious -- I wonder why this is...

# UN
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
UN_chisq1 = kilgariff_chisq(df_corp_auth = version_2s,  
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

print("UN vs. ML published up to 2016: " + str(UN_chisq) + ", UN vs. ML2024: " + str(UN_chisq1))
print("UN vs. ML published up to 2016: " + str(round(UN_chisq)) + ", UN vs. ML2024: " + str(round(UN_chisq1))) # curious -- I wonder why this is...


# US
US_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)
US_chisq1 = kilgariff_chisq(df_corp_auth = version_2s,  
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("US vs. ML published up to 2016: " + str(US_chisq) + ", US vs. ML2024: " + str(US_chisq1))
print("US vs. ML published up to 2016: " + str(round(US_chisq)) + ", US vs. ML2024: " + str(round(US_chisq1))) # curious -- I wonder why this is...



# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2024 ML blurbs SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2012-2016 published ML corpora SPECIFICALLY
# =============================================================================
# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# Vary n (i.e. how many top words)
# # =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# Without stopwords and numbers
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3s, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3s, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the two ML corpora
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
EU_chisq1 = kilgariff_chisq(df_corp_auth = version_2s,  
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)

print("EU vs. ML published up to 2016: " + str(EU_chisq) + ", EU vs. ML2024: " + str(EU_chisq1))
print("EU vs. ML published up to 2016: " + str(round(EU_chisq)) + ", EU vs. ML2024: " + str(round(EU_chisq1))) # curious -- I wonder why this is...

# UN
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
UN_chisq1 = kilgariff_chisq(df_corp_auth = version_2s,  
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)

print("UN vs. ML published up to 2016: " + str(UN_chisq) + ", UN vs. ML2024: " + str(UN_chisq1))
print("UN vs. ML published up to 2016: " + str(round(UN_chisq)) + ", UN vs. ML2024: " + str(round(UN_chisq1))) # curious -- I wonder why this is...


# US
US_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)
US_chisq1 = kilgariff_chisq(df_corp_auth = version_2s,  
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("US vs. ML published up to 2016: " + str(US_chisq) + ", US vs. ML2024: " + str(US_chisq1))
print("US vs. ML published up to 2016: " + str(round(US_chisq)) + ", US vs. ML2024: " + str(round(US_chisq1))) # curious -- I wonder why this is...


# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2012-2016 published ML corpora SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2024 ML blurbs SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


































# RESULT: WHICH ARE MORE SIMILAR (RELATIVELY) TO THE TECHNICAL TEXTS--of EU/US/UN--AND WHICH ARE THEY MORE SIMILAR TO (causal/recent vs. published/old)

'''
'''








