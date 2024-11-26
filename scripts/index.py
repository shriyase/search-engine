import math
import numpy as np


def create_inverted_index(data):
    """
    Create inverted index and calculate the TF-IDF for the documents.
    """
    terms = set(word for doc in data["text"] for word in doc)
    num_docs = len(data)
    inv_index = {}

    for term in terms:
        doc_list = [i for i, doc in enumerate(data["text"]) if term in doc]
        inv_index[term] = [len(doc_list), doc_list]

    tf_idf_docs = []
    for doc_id in range(num_docs):
        weight_vec = []
        for term in terms:
            tf = sum(1 for word in data["text"][doc_id] if word == term)
            idf = math.log(num_docs / (1 + len(inv_index[term][1])), 10)
            weight_vec.append((1 + math.log(tf, 10)) * idf if tf > 0 else 0)
        tf_idf_docs.append(weight_vec)

    return inv_index, tf_idf_docs, terms
