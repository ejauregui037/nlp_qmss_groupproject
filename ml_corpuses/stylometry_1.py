#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 10:53:34 2025

@author: elenafj
"""

# =============================================================================
# Potential methods for stylometry
# =============================================================================
# Load data
import pandas as pd

# technical corpuses (ML)
mlpapersummaries = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_papersummaries.csv")
mlcorps = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_corpuses.csv")
mlcorps_noref = pd.read_csv("~/Desktop/Columbia/courses/NLP_QMSS/nlp_qmss_groupproject/ml_corpuses/ml_corpuses_noref.csv")

# legal data
legal_dat = pd.read_pickle("~/Downloads/cleaned_df.pkl") # trying out Haley's data -- change location

# =============================================================================
# Clean scraped legal texts: very messy right now
# =============================================================================
print(legal_dat.iloc[1,1])
# It's very messy, but it's not clear to me that I can clean it in useful ways. 
# I will first run the scripts/functions on these corpuses as they are

# =============================================================================
# Decisions
# =============================================================================

# Decide which data to use 
# & which parts to analyze for stylometry

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

""" 
Begin stylometry with those author designations.
"""
# Who are the authors we are analyzing?
# I can just cite these in a loop, as the colum in the df

# Lowercase the tokens so that the same word, capitalized or not, counts as one word
version_1['corpuses'] = [corpus.lower() for corpus in version_1['corpuses']]
version_2['corpuses'] = [corpus.lower() for corpus in version_2['corpuses']]
version_3['corpuses'] = [corpus.lower() for corpus in version_3['corpuses']]
version_4['corpuses'] = [corpus.lower() for corpus in version_4['corpuses']]


# Write a function to...
import nltk
'''
Thoughts/concerns: should there be another subfunction where I am comparing them to each other one by one?
Right now, it's checking the PROB (chisq) that all "authors" are from the same distribution, this joint
But I think that what I want is the probability that, for example, the legal documents came from the JOINT
of the TECHNICAL documents... and then that they each came from each other?? am i trying to contrast them 
as well? i think that would be informative...
'''

"""
@ elena: DO THE ABOVE as a diff set of analyses! I would say those have diff INTERPRETATIONS! get this to work first...
Also: reading through this code, it's all about WORD COUNT... what if I want something a bit more complex? In terms of PHRASING
(!) and how they are USING the diff words, phrases they use, etc? Look for methods...
"""

# Calculate chisquared for each of the two candidate authors
def kilgariff_chisq(df_corp_auth):
    '''
    # Take the corpora associated with two authors.
    # Merge them into a single, larger corpus.
    # Count the tokens for each of the words that can be found in this larger corpus.
    # Select the n most common words in the larger corpus.
    # Calculate how many tokens of these n most common words we would have expected to find in each of the two original corpora if they had come from the same author. 
    # Calculate a chi-squared distance by summing, over the n most common words, the squares of the differences between the actual numbers of tokens found in each author’s corpus and the expected numbers, divided by the expected numbers. 
        # lower chi-sq = more similar; bigger chi-sq = lower p-value = less likely to have occurred 'by chance' [under the null dist/hypothesis that they arose from the same distribution/overall corpus]
    '''
    
    # Take the corpora associated with two authors. -- Merge them into a single, larger corpus.
        # note: I want to generalize this to > 1 author
        # (!) note: how strongly will this be affected by the messiness of the current legal document text corpora?
    joint_corp = " ".join(df_corp_auth['corpuses'].tolist())
    
    # Count the tokens for each of the words that can be found in this larger corpus.
    joint_freq_dist =  nltk.FreqDist(joint_corp.split())
    
    # Select the n most common words in the larger corpus.
    most_common = list(joint_freq_dist.most_common(500)) # n = 500; could be an input to the fxn
    print(most_common[1:5]) # INTERNALS
    
    for author in df_corp_auth['author'].unique():
        # Calculate how many tokens of these n most common words we would have expected to find in each of the two original corpora if they had come from the same author. 
        df_authorsubset = df_corp_auth[df_corp_auth['author'] == author] # iterating through authors above
        author_corp = " ".join(df_authorsubset['corpuses'].tolist())
        author_share = (len(author_corp) # of their TOKENS?? confused... i feel like I have not set up my dfs for this...
                        / len(joint_corp)) 
        
    # Calculate a chi-squared distance by summing, over the n most common words, the squares of the differences between the actual numbers of tokens found in each author’s corpus and the expected numbers, divided by the expected numbers. 
    '''
    My approach: "What is the probability that the legal text could have been drawn from the ML corpus?"
    Mathematically: sum( ((Ci-Ei)^2)/Ei ), where Ei is derived by getting the frequency in the joint ML corpus. 
    I think that makes the most sense
    '''
    # question
    '''
    Question: should we be clearning out stop words? I feel like no, bc I want a sense for how they are using these words in tandem...
    But then--doesn't that mean that maybe I should be extending this method to BIGRAMS and TRIGRAMS??
    I COULD... (!)
    '''
    # lower chi-sq = more similar; bigger chi-sq = lower p-value = less likely to have occurred 'by chance' [under the null dist/hypothesis that they arose from the same distribution/overall corpus]
        chisquared = 0
        for word, joint_count in most_common:

            # How often do we really see this common word? 
            # -- @elena: I tried to translate this from the example code to my goals as clearly as I could, but I'm not convinced that this is "correct"
            overall_count = joint_corp.count(word) # overall -- how is this difffrom joint_count??
            author_count = author_corp.count(word) # legal docs

            # How often should we see it?
            expected_author_count = joint_count * (1-author_share)  
            # I flipped their proportions, cause i think for me "author" (legal) is conceptually what they are using as "disputed" author...
            expected_disputed_count = joint_count * (author_share) 
            # uhhhh is it?? does this apply even when there is >1 author?

            # Add the word's contribution to the chi-squared statistic
            chisquared += ((overall_count-expected_author_count) *
                            (overall_count-expected_author_count) /
                            expected_author_count)

            chisquared += ((disputed_count-expected_disputed_count) *
                            (disputed_count-expected_disputed_count)
                            / expected_disputed_count)

        print("The Chi-squared statistic for candidate", author, "is", chisquared)
        
    

# Now: apply it to each of the four "versions"
kilgariff_chisq(version_1)
# LMAO
# THESE ARE FUCKING HUGE
# okay something is CLEARLY wrong
# I will troubleshoot and go through step by step after lunch!!!!! & do some "in-between" printing to sort of 
# troubleshoot and understand intermediate outputs!!!

# =============================================================================
# Go 2
# =============================================================================
'''
https://fastdatascience.com/natural-language-processing/fast-stylometry-python-library/
However, this library uses the Delta method and has some PCA analyses built in which I think
would be nice visuals!! of how similar the various regulatory bodies are in space, compared to the ML
corpuses (can set those as a "single author")

Note: for each of these, make sure to output and visualize the top words from each corpus, ala
word frequency that Karina is doing. (That way we will know what is *drivng* these results & it'll
be a bit more interpretable.')
'''

'''
Note: this is probably a semantic analysis, but could we compare the embeddings of key tokens 
to see how similar they are between the bodies of text? e.g. 1) embedding of "machine learning"
with Word2Vec, from the body of technical texts, then 2) doing that for the regulatory texts, grouped 
by nation/region; then 3) calculating "distance" between those embeddings, and 4) as a demonstration:
finding the "distance" from the same embedding generated over a *different* set of academic ML texts?
    AJA!! https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
    Semantic textual similarity
    https://christianbernecker.medium.com/nlp-similarity-use-pretrained-word-embeddings-for-semantic-similarity-search-with-bert-4beaf7b6a148 ?
    https://www.pingcap.com/article/top-10-tools-for-calculating-semantic-similarity/
'''

# =============================================================================
# GO 3
# =============================================================================
'''    
Other useful sources/thoughts:
https://www.reddit.com/r/MLQuestions/comments/t7k3gi/is_it_possible_to_compute_the_average_semantic/
https://medium.com/@swarup.t/exploring-contextual-text-similarity-a-dive-into-machine-learning-techniques-3d477c88bf20
https://datascience.stackexchange.com/questions/71512/nlp-simple-approach-to-identify-commonalities-in-text-comments-between-people
https://medium.com/@evertongomede/exploring-the-depths-of-meaning-semantic-similarity-in-natural-language-processing-19281e58558e
https://www.geeksforgeeks.org/different-techniques-for-sentence-semantic-similarity-in-nlp/#
https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681/

https://medium.com/@rahultiwari065/ultimate-guide-to-text-similarity-from-basics-to-advanced-applications-1492f82c0269
    ** patterns and sequences between texts -- phrases?? -- Ratcliff-Obershelp Algorithm

https://www.index.dev/blog/best-nlp-algorithms-to-get-document-similarity
    Interesting ideas here -- esp. skip gram
    
stylo() in R! seems VERY useful!! and user-friendly
https://guides.temple.edu/stylometryfordh/programs
https://computationalstylistics.github.io/resources/
'''

# =============================================================================
# end of file
# =============================================================================


