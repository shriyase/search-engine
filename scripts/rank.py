import numpy as np
from numpy.linalg import norm
import pandas as pd
import math


def rank_documents(query, inv_index, tf_idf_docs, terms, data):
    """
    Rank documents based on their similarity to the query.
    """
    query_terms = tokenize_and_normalize(pd.DataFrame({"text": [query]}))["text"][0]
    query_vector = [0] * len(terms)

    for i, term in enumerate(terms):
        tf = query_terms.count(term)
        idf = inv_index[term][0]
        query_vector[i] = (1 + math.log(tf, 10)) * idf if tf > 0 else 0

    query_vector = np.array(query_vector)
    norm_query = norm(query_vector)

    ranking = {}
    for doc_id, doc_vector in enumerate(tf_idf_docs):
        cosine_similarity = np.dot(query_vector, doc_vector) / (
            norm_query * norm(doc_vector)
        )
        ranking[doc_id] = cosine_similarity

    ranked_docs = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    return [(data.iloc[doc[0]]["text"], doc[1]) for doc in ranked_docs]
