#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:38:42 2025

@author: elenafj
"""

'''
Stylometric Analysis #8

Textual entailment: 
    is a task where the goal is to determine if one text 
    logically follows from or is entailed by another text. 
Paraphrase detection: 
    looks at whether two texts express the same idea in different ways.
'''

# This is a really interesting set of approaches, but I must define my 
# research questions VERY specifically in order to make them useful

'''
Specific research questions
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
