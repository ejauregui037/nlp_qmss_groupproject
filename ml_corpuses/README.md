# ml_technical_researchpublications/

This directory contains the code used to generate our 'technical' ML corpuses, i.e. ML publications in academia. 

This folder contains the .py used to generate the csvs below, as well as the csvs themselves. In future scripts, these csvs can be loaded and cleaned to the degree appropriate for various analyses. Each row of the csvs below is a single corpus (i.e. a single paper).

In analyses, I would propose that Set 1 & Set 2 should be used together. 

*In the future, I might throw another .py file in this folder where I do just that^^ i.e. merge the dataframes and rewrite that to a new csv.*

### Set 1
- ml_corpuses.csv : Contains corpuses up to 2016 from a git page which had compiled key publications across different ML areas. 
- ml_corpuses_noref.csv : Contains the same corpuses as ml_corpuses.csv, but here they have been cleaned to remove all text which appears after the token (string) "References" -- that is, to exclude Reference lists at the end of the publication (since that is not substantive content within the paper). Note that these corpuses *include* the word "References" itself.

### Set 2
- ml_recent.csv :
- ml_recent_norefs.csv :
