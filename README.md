This Python script helps you analyze two sets of text data (like technical papers, abstracts, or articles) by comparing:
1. How many times each word or phrase appears (Frequency Count)
2. How important each word or phrase is in each corpus (TF-IDF)

 It supports n-grams (combinations of 1 to 3 words), so it can catch things like “machine learning" or "neural network" and not just single words.   
 
 What Is This Useful For?
 - Comparing research trends between two groups (e.g., authors from different regions).
 - Identifying unique or shared vocabulary between corpora.
 - Highlighting key concepts in technical papers or articles.  
What the Script Does?
  Step 1: Word/N-gram Frequency
Counts how many times each word or phrase appears in: - Corpus 1 (your first DataFrame) - Corpus 2 (your second DataFrame)  It removes common filler phrases like `"based on"` or `"in this paper"` to give cleaner results.

  Step 2: TF-IDF Comparison
  Calculates TF-IDF scores for each phrase, which represent how important that phrase is in one corpus compared to the other.  It shows which words are more distinctive in one set of texts than the other. 
