"""
Query Expander code
Advanced DB Proj 1
Team members:-
Paras Tehria(pt2557), Sakshi Arora(sa3871)

"""

import sys
import json
import math
import requests
import pandas as pd
import nltk
from nltk import word_tokenize
from collections import defaultdict

nltk.download('punkt')

stop_words = set(pd.read_csv("stopwords.csv", header=None).iloc[:, 0].tolist())

ALPHA = 1
BETA = 0.85
GAMMA = 0.25


def get_search_results(curr_query):
    """
    Returns a list of top 10 search results

    :param curr_query:
    :return:
    """
    search_results = fetch_google_search_api(curr_query)

    for idx, el in enumerate(search_results):
        print("Result {}\n[\nURL: {}\nTitle: {}\nSummary: {}\n]".format(str(idx + 1), el['link'],
                                                                        el['title'], el['snippet']))

        input_feedback = input("Relevant(Y/N)?")
        if input_feedback.upper() == "Y":
            el['relevant'] = True
        else:
            el['relevant'] = False

    return search_results


def fetch_google_search_api(curr_query):
    """
    Uses google api to fetch top 10 query results
    :param curr_query:
    :return:
    """
    print("Parameters:")
    print("Client key = {}\nEngine key = {}\nQuery = {}\nPrecision = {}".format(custom_search_api, search_engine_id,
                                                                                curr_query, str(target_prec)))

    # Google search
    url = "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}".format(custom_search_api,
                                                                                search_engine_id, curr_query)
    response = requests.get(url)
    search_results = json.loads(response.text)['items']

    results = []
    for item in search_results[:10]:
        results.append({"title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")})

    return results


def expand_query(search_results, curr_query, curr_prec):
    """
    Expands query using rocchio's algorithm

    :param search_results:
    :param curr_query:
    :param curr_prec:
    :return:
    """

    token_search_idx, freq_list = get_indexed_ls(search_results)

    word_weights = get_word_weights(token_search_idx, freq_list, search_results, curr_query)

    original_query = curr_query.split(" ")
    sorted_key_weights = {k: v for k, v in sorted(word_weights.items(), key=lambda item: item[1], reverse=True)}

    new_words = []
    count = 0
    for k, v in sorted_key_weights.items():
        if k not in original_query:
            new_words.append(k)
            count += 1
        if count == 2:
            break

    print("Query: {} ".format(curr_query))
    print("Precision: {}".format(str(curr_prec)))

    if curr_prec < target_prec:
        print("Still below the desired precision of " + str(target_prec))
        words = " " + " ".join(new_words)
        print("Augmenting by " + words)
        curr_query += words
    else:
        print("Desired precision reached, done")

    return curr_query


def get_indexed_ls(search_results):
    """
    Get indexed lists required for Rocchio algorithm
    :param search_results:
    :return:
    """
    token_search_idx = defaultdict(set)
    freq_list = [defaultdict(int)] * len(search_results)

    for idx, el in enumerate(search_results):
        tokens = word_tokenize(el["snippet"]) + word_tokenize(el["title"])

        modified_tokens = []
        for w in tokens:
            if w.lower() not in stop_words and w.isalpha():
                modified_tokens.append(w.lower())

        for token in modified_tokens:
            freq_list[0][token] += 1
            token_search_idx[token].add(idx)

    return token_search_idx, freq_list


def get_word_weights(token_search_idx, freq_list, search_results, curr_query):
    """
    Implements rocchio algorithm for query expansion

    Citation:- https://nlp.stanford.edu/IR-book/pdf/09expand.pdf

    :param token_search_idx:
    :param freq_list:
    :param search_results:
    :param curr_query:
    :return:
    """

    new_words_weights = defaultdict(float)
    term_weights = defaultdict(int)

    relevant = defaultdict(int)
    nonrelevant = defaultdict(int)

    rel_count = 0
    count_non_rel = 0

    for idx, el in enumerate(search_results):
        if el["relevant"]:
            rel_count += 1
            for k, v in freq_list[idx].items():
                relevant[k] += v
        else:
            count_non_rel += 1
            for k, v in freq_list[idx].items():
                nonrelevant[k] += v

    for word in curr_query.split(" "):
        new_words_weights[word] = 1.0

    for k, v in token_search_idx.items():
        idf = math.log10(float(len(search_results)) / float(len(v)))

        for idx in v:
            if search_results[idx]['relevant']:
                term_weights[k] += BETA * idf * (float(relevant[k]) / rel_count)
            else:
                term_weights[k] -= GAMMA * idf * (float(nonrelevant[k]) / count_non_rel)

        if k in new_words_weights:
            new_words_weights[k] = ALPHA * new_words_weights[k] + term_weights[k]
        elif term_weights[k] > 0:
            new_words_weights[k] = term_weights[k]

    return new_words_weights


def query_iterator(curr_query):
    """
    Iterates over queries until we reach required precision
    :param curr_query:
    :return:
    """
    iteration = 0
    curr_prec = 0.0
    while 0.0 < curr_prec < target_prec or iteration == 0:
        print("Iteration no. {}".format(iteration))

        # Find search result for current iteration
        search_results = get_search_results(curr_query)

        # Expand query based on current precision
        curr_prec = float(sum([1.0 if x['relevant'] else 0.0 for x in search_results]) / len(search_results))
        curr_query = expand_query(search_results, curr_query, curr_prec)

        iteration += 1

    return


if __name__ == '__main__':
    inputs = sys.argv

    print(inputs)

    custom_search_api, search_engine_id, target_prec, query = inputs[1], inputs[2], float(inputs[3]), " ".join(
        inputs[4:])
    query_iterator(query)
