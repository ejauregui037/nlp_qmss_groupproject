#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 16:54:58 2025

@author: elenafj
"""

import sys 
import requests
import bs4 
import PyPDF2 
import pdfminer 
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup 
import pandas as pd
import io


"""
Step 1 of project: 
    Scrape top ML papers to have a base from which to compare the syntax, semantics, etc. to the 
    regulatory documents.
"""

"""
First source: https://github.com/terryum/awesome-deep-learning-papers?tab=readme-ov-file#natural-language-processing--rnns
Note: only maintinaed up to 2016. Other source required to supplement with more recent papers 
(since the field is exploding in real time.)
"""

# use beautifulsoup to iterate through embedded PDFs and access text/html from each 

# =============================================================================
# Historical key/representative ML texts 
# =============================================================================

# Soupify URL
my_url = "https://github.com/terryum/awesome-deep-learning-papers?tab=readme-ov-file#natural-language-processing--rnns"
result = requests.get(my_url)
src = result.content
page_soup = soup(src, "lxml") # soup for full webpage


type(page_soup) # it is a bs4.Beautiful Soup object
print(page_soup.prettify()) # prints html prettily
page_soup.title
page_soup.name
page_soup.get_text() # text only -- will be useful for the pdfs...

pdf_links = []
for link in page_soup.find_all('a'): # find all links -- note: this INCUDES the [pdf] links... right?
    print(link.get('href'))
    pdf_links.append(link.get('href')) # len = 416
    # note: some links work, some are broken...
    # Limit to those with "pdf" in the link? -- I think that's a fine heuristic here
    
# limit to those with "pdf" in the link name
type(pdf_links[1]) # string
# filter list of strings ala grep
pdf_links = [s for s in pdf_links if "pdf" in s] # 211 -- check: that's about right!! on the webpage, 217 results for Find: "[pdf]" : )

# check
# pdf_links[1]    # nice

# Now: iterate through links -- get pdf for each, then save into a set of strings
# for the glitchy ones:
from PyPDF2.errors import PdfReadError  # Import the exception from the correct module

# my function
def get_article_text(list_o_pdflinks):
    """

    Parameters
    ----------
    list_o_pdflinks : list
        Each element in the list must a string -- specifically, a string that is an html link to open a pdf.

    Returns
    -------
    List of scraped PDF text (References included), where each element in the list is the text for all pages in the pdf, compressed into a single string object.

    """
    list_of_strings = []
    ii = 0
    for link_ii in list_o_pdflinks: # iterate through list of pdf links
        print(ii) # tracking the loop -- for when it breaks with PdfREadErrors... also: broadly nice to know how quickly the function/program is running. Also: this way, can eliminate PDFs with errors that seem unsystematic/unnecessarily buggy

        # Run only for the PDFs (links) that are NOT buggy for unsystematic reasons -- e.g. PdfReadError: EOF marker not found.... I am not going to deal with nitpicky java errors like that right now.
        if (ii not in [50,80,106,112,113,119,120,121,123,124,125,206,207]) : # run only if they are NOT the buggy ones, listed here (=13)
            try:
                # Extract text with requests + PyPDF2
                r = requests.get(link_ii)
                f = io.BytesIO(r.content)
                reader = PyPDF2.PdfReader(f)
                # Loop through all pages and extract the text
                all_contents = []
                for page_num in range(len(reader.pages)):  # Iterate over all pages
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    text = text.split('\n')  # splits & removes "\n" from text
                    text = " ".join(text) 
                    all_contents.append(text)  # all_contents : becomes combined string for each page (list with len = # pages in pdf)
                all_contents_str = " ".join(all_contents)  # now we have one string per pdf link
                list_of_strings.append(all_contents_str)
            except PdfReadError as e: # there are roughly 18 (1:150) + 12 = 30
                print(f"Raised PdfReadError: {e}")
        ii += 1 # keep the count no matter whether it runs or not
    # return the list of strings: 1 per PDF link
    return(list_of_strings)

# Run function on our list of links
# Eliminate the PDFs (links) that are buggy for unsystematic reasons -- e.g. PdfReadError: EOF marker not found.... I am not going to deal with nitpicky java errors like that right now.
# test = get_article_text(pdf_links[0:10])  # note runtime--LONG. Just an excruciating process... Break it up into chunks to debug? In case one of the pfs is broken or some such?
list_ml_corpuses = get_article_text(pdf_links)

# Now: iterate through the complete strings to remove text after "References"
def cut_string_after_keyword(text, keyword):
    """Cuts off a string after the first occurrence of a keyword, including the keyword.

    Args:
        text: The string to cut.
        keyword: The keyword to search for.

    Returns:
        The string cut off after the keyword, or the original string if the keyword is not found.
    """
    try:
      index = text.index(keyword)
      return text[:index + len(keyword)]
    except ValueError:
        return text
    
# cut after "References"
list_ml_corpuses_noref = list_ml_corpuses.copy()
for ii in range(len(list_ml_corpuses_noref)):
    list_ml_corpuses_noref[ii] = cut_string_after_keyword(list_ml_corpuses_noref[ii], "References")

# store the list of corpuses (no refs) in a pandas df and export/write to csv
ml_corpuses = pd.DataFrame(list_ml_corpuses)
ml_corpuses_noref = pd.DataFrame(list_ml_corpuses_noref)



# export to csv
ml_corpuses.to_csv("/Users/elenafj/Desktop/Columbia/courses/NLP_QMSS/ml_corpuses.csv", index=False, encoding = "utf-8", errors = "replace")
ml_corpuses_noref.to_csv("/Users/elenafj/Desktop/Columbia/courses/NLP_QMSS/ml_corpuses_noref.csv", index=False, encoding = "utf-8", errors = "replace")

# =============================================================================
# More modern ML texts (2024) -- to incorporate updates in tech, which moves @ 1e6 MPH 
# =============================================================================


list_ml_corpuses_noref[9]








# =============================================================================
# end of file
# =============================================================================







# Note: these are up to 2016. Also scrape PDFs for list of recent publications... **





r = requests.get(pdf_links[1] )
f = io.BytesIO(r.content)
reader = PyPDF2.PdfReader(f)
# Loop through all pages and extract the text
all_contents = []
for page_num in range(len(reader.pages)):  # Iterate over all pages
    page = reader.pages[page_num]
    text = page.extract_text()
    text = text.split('\n')
    text = " ".join(text)
    all_contents.append(text)
    
len(all_contents) # 20 pages
all_contents[13]

# then I think we should clean this to...
# a) exclude stopwords & punctuation?... but we CAN'T if we want to analyze syntax, etc
# b) exclude numbers
# Note: for syntax & semantics, really the ONLY thing I want to exclude that I feel strongly about are the refernces
# But that is just... so annoying LOL

all_contents_str = " ".join(all_contents)

def cut_string_after_keyword(text, keyword):
    """Cuts off a string after the first occurrence of a keyword, including the keyword.

    Args:
        text: The string to cut.
        keyword: The keyword to search for.

    Returns:
        The string cut off after the keyword, or the original string if the keyword is not found.
    """
    try:
      index = text.index(keyword)
      return text[:index + len(keyword)]
    except ValueError:
        return text









# now: scrape these for their content
list_ml_corpuses = list()
for link in pdf_links[0:1]:
    result = requests.get(pdf_links[0])
    pdf_response = requests.get(pdf_links[0], stream=True)
    page_soup = soup(src, "lxml")
    ml_texts = page_soup.get_text()
    list_ml_corpuses.append(ml_texts)
    
# then turn this list into a df of corpuses...
    