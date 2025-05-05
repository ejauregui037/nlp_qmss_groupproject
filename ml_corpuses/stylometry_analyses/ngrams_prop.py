#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:34:15 2025

@author: elenafj
"""

"""
Stylometric Analysis #3

N-Grams: 
    Looks at sequences of N words (bigrams, trigrams, etc.). This 
    can help identify recurring patterns or phrases, showing how 
    authors tend to link ideas together or phrase things in a 
    characteristic way.
    
Use Kilgariff's Chi-Squared test as a 'goodness of fit' test, 
where the ML texts are the underlying 'gold standard' distribution
for the degree of 'technicality'.

***BUT: use PROPORTIONAL chi-sq, so that the major imbalance between the expected 
frequencies comparing legal corpuses does not influence our results!!**
"""

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
# extract the necessary dfs/columns -- we will focus on a) version without numbers & b) version without numbers + stopwords
# =============================================================================
print((version_2n123[2]))

# bigrams
version_2n_bigrams = version_2n123[1]
version_2s_bigrams = version_2s123[1]
version_3n_bigrams = version_3n123[1]
version_3s_bigrams = version_3s123[1]

# trigrams
version_2n_trigrams = version_2n123[2]
version_2s_trigrams = version_2s123[2]
version_3n_trigrams = version_3n123[2]
version_3s_trigrams = version_3s123[2]


# figuring out how to flatten these lists within the function... for tuples
# test = list(version_3s_trigrams[version_3s_trigrams['author']=="US"]['ngram1'])
# test = [item for sublist in version_3s_trigrams[version_3s_trigrams['author'] == "US"]['ngram1'] for item in sublist]
# test = test[0]
# nltk.FreqDist(test)

# =============================================================================
# 
# =============================================================================

# Karina is already doing this in a basic Count way

# Kilgariff chi-sq, but now applied to ngrams:
    
import nltk

# we already did ngrams of length 1 (implicitly) in the previous script! 
# functions below are for bigrams and trigrams  (note: column is named "ngram1" regardless)


# Calculate chisquared for each of the two candidate authors 
# -- but now the "tokens" are ngrams...
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
    # refML_dist =  " ".join(df_corp_auth[df_corp_auth['author']==ref_list]['ngram1'].tolist())  # this is the name of the bigram/trigram column
    refML_dist = [item for sublist in df_corp_auth[df_corp_auth['author'] == ref_list]['ngram1'] for item in sublist]

    # Count the tokens for each of the words that can be found in this larger corpus.
    refML_freq_dist =  nltk.FreqDist(refML_dist)
    
    # and the total number of tokens, to get frequency
    refML_totaltokens = sum(refML_freq_dist.values())
    
    if internals == True: 
        print(refML_totaltokens) # INTERNALS
    
    # Select the n most common words in the larger corpus.
    refML_most_common = list(refML_freq_dist.most_common(n_mostcomm)) # n = 500; could be an input to the fxn
    
    if internals == True: 
        print(refML_most_common[1:20]) # INTERNALS
        
    # make a joint corpus for each 'comparison' group of the legal
    # compar_dist =  " ".join(df_corp_auth[df_corp_auth['author']==compar_grp]['corpuses'].tolist()) 
    compar_dist = [item for sublist in df_corp_auth[df_corp_auth['author'] == compar_grp]['ngram1'] for item in sublist]
    # and the total number of tokens, to get frequency
    compar_freq_dist =  nltk.FreqDist(compar_dist)
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
        
        # make them proportions -- keeping name as "counts" for simplicity ***********
        expected_compar_count = expected_compar_count/compar_totaltokens 
        observed_compar_count = observed_compar_count/compar_totaltokens # legal docs
        
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
# =============================================================================
# =============================================================================
# BIGRAMS
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Without numbers -- didn't run this for bigrams because the top 20, for example, in the ref ML were:
# [(('in', 'the'), 2321), (('on', 'the'), 1576), (('to', 'the'), 1487), (('et', 'al.'), 1015), (('can', 'be'), 951), (('for', 'the'), 856), (('et', 'al'), 853), (('and', 'the'), 849), (('from', 'the'), 775), (('number', 'of'), 708), (('neural', 'networks'), 673), (('of', 'a'), 663), (('with', 'the'), 650), (('in', 'a'), 645), (('with', 'a'), 641), (('as', 'a'), 635), (('in', 'this'), 625), (('such', 'as'), 597), (('is', 'a'), 585)]
# =============================================================================

# EU_chisq = kilgariff_chisq(df_corp_auth = version_3n_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'EU', 
#                 n_mostcomm = 500, 
#                 internals = True)

# UN_chisq = kilgariff_chisq(df_corp_auth = version_3n_bigrams, 
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'UN', 
#                 n_mostcomm = 500, 
#                 internals = True)

# US_chisq = kilgariff_chisq(df_corp_auth = version_3n_bigrams, 
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'US', 
#                 n_mostcomm = 500, 
#                 internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

# print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
# print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without stopwords and numbers
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3s_bigrams, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3s_bigrams, 
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
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
EU_chisq1 = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

print("EU vs. ML published up to 2016: " + str(EU_chisq) + ", EU vs. ML2024: " + str(EU_chisq1))
print("EU vs. ML published up to 2016: " + str(round(EU_chisq)) + ", EU vs. ML2024: " + str(round(EU_chisq1))) # curious -- I wonder why this is...

# UN
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
UN_chisq1 = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

print("UN vs. ML published up to 2016: " + str(UN_chisq) + ", UN vs. ML2024: " + str(UN_chisq1))
print("UN vs. ML published up to 2016: " + str(round(UN_chisq)) + ", UN vs. ML2024: " + str(round(UN_chisq1))) # curious -- I wonder why this is...


# US
US_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)
US_chisq1 = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("US vs. ML published up to 2016: " + str(US_chisq) + ", US vs. ML2024: " + str(US_chisq1))
print("US vs. ML published up to 2016: " + str(round(US_chisq)) + ", US vs. ML2024: " + str(round(US_chisq1))) # curious -- I wonder why this is...


# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2024 blurbs published ML corpora SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2024 ML blurbs SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================------------------------------
# Vary n (i.e. how many top words)
# # =============================================================================----------------------------

# =============================================================================
# Without stopwords and numbers
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3s_bigrams, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3s_bigrams, 
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
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
EU_chisq1 = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)

print("EU vs. ML published up to 2016: " + str(EU_chisq) + ", EU vs. ML2024: " + str(EU_chisq1))
print("EU vs. ML published up to 2016: " + str(round(EU_chisq)) + ", EU vs. ML2024: " + str(round(EU_chisq1))) # curious -- I wonder why this is...

# UN
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
UN_chisq1 = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)

print("UN vs. ML published up to 2016: " + str(UN_chisq) + ", UN vs. ML2024: " + str(UN_chisq1))
print("UN vs. ML published up to 2016: " + str(round(UN_chisq)) + ", UN vs. ML2024: " + str(round(UN_chisq1))) # curious -- I wonder why this is...


# US
US_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)
US_chisq1 = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("US vs. ML published up to 2016: " + str(US_chisq) + ", US vs. ML2024: " + str(US_chisq1))
print("US vs. ML published up to 2016: " + str(round(US_chisq)) + ", US vs. ML2024: " + str(round(US_chisq1))) # curious -- I wonder why this is...

# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2024 ML blurbs SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without stopwords and numbers -- Compare the various legislative bodies to the 2012-2016 published ML corpora SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2s_bigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))



# =============================================================================
# =============================================================================
# =============================================================================
# TRIGRAMS
# =============================================================================
# =============================================================================
# =============================================================================


# =============================================================================
# Without numbers --really useful/interesting for trigrams to actually LEAVE stopwords in!!
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3n_trigrams,  # lots of stopwords -- i don't think it's informative!
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)  # punctuation is far too common in the tuples... they MUST be removed. -- and i did remove them!! : ) modified cleandata_py : )

UN_chisq = kilgariff_chisq(df_corp_auth = version_3n_trigrams, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3n_trigrams, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# # =============================================================================
# # Without stopwords and numbers
# # =============================================================================

# EU_chisq = kilgariff_chisq(df_corp_auth = version_3s_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'EU', 
#                 n_mostcomm = 500, 
#                 internals = True)

# UN_chisq = kilgariff_chisq(df_corp_auth = version_3s_trigrams, 
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'UN', 
#                 n_mostcomm = 500, 
#                 internals = True)

# US_chisq = kilgariff_chisq(df_corp_auth = version_3s_trigrams, 
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'US', 
#                 n_mostcomm = 500, 
#                 internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

# print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
# print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without numbers & WITH stopwords -- Compare the various legislative bodies to the two ML corpora
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
EU_chisq1 = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)

print("EU vs. ML published up to 2016: " + str(EU_chisq) + ", EU vs. ML2024: " + str(EU_chisq1))
print("EU vs. ML published up to 2016: " + str(round(EU_chisq)) + ", EU vs. ML2024: " + str(round(EU_chisq1))) # curious -- I wonder why this is...

# UN
UN_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
UN_chisq1 = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)

print("UN vs. ML published up to 2016: " + str(UN_chisq) + ", UN vs. ML2024: " + str(UN_chisq1))
print("UN vs. ML published up to 2016: " + str(round(UN_chisq)) + ", UN vs. ML2024: " + str(round(UN_chisq1))) # curious -- I wonder why this is...


# US
US_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)
US_chisq1 = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("US vs. ML published up to 2016: " + str(US_chisq) + ", US vs. ML2024: " + str(US_chisq1))
print("US vs. ML published up to 2016: " + str(round(US_chisq)) + ", US vs. ML2024: " + str(round(US_chisq1))) # curious -- I wonder why this is...

# =============================================================================
# Without numbers & WITH stopwords -- Compare the various legislative bodies to the 2024 ML blurbs SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================
# Without numbers & WITH stopwords -- Compare the various legislative bodies to the 2012-2016 published ML corpora SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 500, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 500, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 500, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))

# =============================================================================------------------------------
# Vary n (i.e. how many top words)
# # =============================================================================----------------------------

# =============================================================================
# Without numbers & WITH stopwords
# =============================================================================

EU_chisq = kilgariff_chisq(df_corp_auth = version_3n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_technical', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)

UN_chisq = kilgariff_chisq(df_corp_auth = version_3n_trigrams, 
                ref_list = 'ml_technical', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)

US_chisq = kilgariff_chisq(df_corp_auth = version_3n_trigrams, 
                ref_list = 'ml_technical', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without numbers & WITH stopwords -- Compare the various legislative bodies to the two ML corpora
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
EU_chisq1 = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)

print("EU vs. ML published up to 2016: " + str(EU_chisq) + ", EU vs. ML2024: " + str(EU_chisq1))
print("EU vs. ML published up to 2016: " + str(round(EU_chisq)) + ", EU vs. ML2024: " + str(round(EU_chisq1))) # curious -- I wonder why this is...

# UN
UN_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
UN_chisq1 = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)

print("UN vs. ML published up to 2016: " + str(UN_chisq) + ", UN vs. ML2024: " + str(UN_chisq1))
print("UN vs. ML published up to 2016: " + str(round(UN_chisq)) + ", UN vs. ML2024: " + str(round(UN_chisq1))) # curious -- I wonder why this is...


# US
US_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)
US_chisq1 = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("US vs. ML published up to 2016: " + str(US_chisq) + ", US vs. ML2024: " + str(US_chisq1))
print("US vs. ML published up to 2016: " + str(round(US_chisq)) + ", US vs. ML2024: " + str(round(US_chisq1))) # curious -- I wonder why this is...


# =============================================================================
# Without numbers & WITH stopwords -- Compare the various legislative bodies to the 2024 ML blurbs SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_2024', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))


# =============================================================================
# Without numbers & WITH stopwords -- Compare the various legislative bodies to the 2012-2016 published ML corpora SPECIFICALLY
# =============================================================================

# EU 
EU_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'EU', 
                n_mostcomm = 50, 
                internals = True)
# UN 
UN_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'UN', 
                n_mostcomm = 50, 
                internals = True)
# US 
US_chisq = kilgariff_chisq(df_corp_auth = version_2n_trigrams,  # bad -- all the top words are symbols! but since we have 500 I feel like these can't be that much of an influence?
                ref_list = 'ml_published', 
                compar_grp = 'US', 
                n_mostcomm = 50, 
                internals = True)

print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))





























# not redone here -- from kilgariff_chisq.py:
    
# =============================================================================
# 'version' ; not cleaned at all [[too messy for bigrams and trigrams]]
# =============================================================================

# EU_chisq = kilgariff_chisq(df_corp_auth = version_3,  # bad -- all the top words are stopwords
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'EU', 
#                 n_mostcomm = 500, 
#                 internals = True)

# UN_chisq = kilgariff_chisq(df_corp_auth = version_3, 
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'UN', 
#                 n_mostcomm = 500, 
#                 internals = True)

# US_chisq = kilgariff_chisq(df_corp_auth = version_3, 
#                 ref_list = 'ml_technical', 
#                 compar_grp = 'US', 
#                 n_mostcomm = 500, 
#                 internals = True) # MOST similar is UN, I believe!! smallest chi-sq value...

# print("EU: " + str(EU_chisq) + ", UN: " +  str(UN_chisq) +  ", US: " + str(US_chisq))
# print("EU: " + str(round(EU_chisq)) + ", UN: " +  str(round(UN_chisq)) +  ", US: " + str(round(US_chisq)))





    
    
    

