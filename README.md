# Information-Retrieval-Rocchio-Algo

Team Members:- Paras Tehria(pt2557), Sakshi Arora(sa3871)

Files Submitting:- query_expander.py, requirements.txt, stopwords.csv, readme.txt

For running:-

-PLEASE USE PYTHON 3.8. Code breaks in python 3.6

First run "pip install -r requirements.txt"
stopwords.csv file should be in the same folder as of the query_expander.py file
running command:- python3 query_expander.py <google api key> <google engine id> <precision> <query>

Project Structure:-

The flow of the project is as follows:-
a.) get search results and feedback from the user using google search api
b.) calculate precision and check if it has reached the desired precision or not
c.) If desired precision is not reached, run the rocchio's algorithm to find new words to expand query with
d.) Continue the process till we reach the desired precision or current precision becomes 0

Query modification method:-
  
We have used Rocchio's algorithm for query expansion. Rocchio's algorithm uses Vector Space Model
with the idea that words in the relevant documents will be closer to the answer document. At each iteration,
we remove some irrelevant document and include some words from relevant documents in the query. ALPHA and BETA params
control the percentage of relevant and irrelevant documents to include.

Citation:- https://nlp.stanford.edu/IR-book/pdf/09expand.pdf

Google Custom Search Engine JSON API Key :- AIzaSyD4cANvEPxUNlRS1ckb4u06Z8ZK1v08gD0
Engine ID :- dc64f97150ea60e6b
