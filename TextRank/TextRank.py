

"""
https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf

use text-8 for vectorization


for each document:
    build graph using the following method:
        for each word in the text:
            find context words (left-right window of size n)
            add undirected edges with weights equal to cosine_similarity

    from graph iteratively compute TextRank scores using above link formula

    Select the top-n (or other selection method) by rank and use THOSE to directly compute the similarity
    (by mean similarity between all? if top 3 for example then 9 comparisons, but not because sum vs sum / 9)


"""

import operator
import os

import gensim.downloader as api
import pandas as pd
from gensim.models import Word2Vec

from preprocessing import *

np.seterr(all="raise")


def single_sim(a, b):
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]


def build_graph(windows, word2vec):
    """
    Aij == weight of edge from vertex i to vertex j
    """
    vocab = set([word for ngram in windows for word in ngram])
    indices = {word: i for i, word in enumerate(vocab)}

    adj = np.zeros(shape=(len(vocab), len(vocab)))

    for ngram in windows:
        for iword in ngram:
            for jword in set(ngram) - set(iword):
                i, j = indices[iword], indices[jword]
                adj[i, j] = np.abs(single_sim(word2vec.wv[iword], word2vec.wv[jword]))
                adj[j, i] = adj[i, j]

    return adj * (1 - np.eye(adj.shape[0])), indices


def build_ranks(graph, indices, d=0.85, max_iter=100, epsilon=1e-4):

    ranks = np.ones(shape=(graph.shape[0])) / len(indices)
    for it in range(max_iter):
        converged = True
        for i in range(graph.shape[0]):  # O(len(vocab))
            err = np.dot(np.reciprocal(np.sum(graph, axis=0) + epsilon), graph[i])
            try:
                new_rank = 1 - d + d * err * ranks[i]
                converged = converged and np.abs(new_rank - ranks[i]) < epsilon
                ranks[i] = new_rank

            except FloatingPointError:
                print(ranks[i], err, graph[i], np.reciprocal(np.sum(graph, axis=0)))
                time.sleep(1)
                exit(0)

        if converged:
            break

    return ranks


def top_n(ranks, n):
    best = []
    try:
        for _ in range(n):
            high_word = max(ranks.items(), key=operator.itemgetter(1))[0]
            best.append(high_word)
            ranks.pop(high_word, None)

        return best
    except ValueError:
        pass
    finally:
        return best


print("Loading model....")


model_name = "with_news.w2v"
if model_name in os.listdir():
    model = Word2Vec.load(model_name)
else:
    text8 = api.load("text8")
    model = Word2Vec(text8)
    model.save("text8.w2v")


print("Loading dataset")

data = pd.read_csv("dataset_repopulated.csv").head(1000)
data["auto_keywords"] = ""
text_raw = list(data["text"])


window_size = 3
stop_eng = stopwords.words("english")

print("Preprocessing dataset")

text_preprocessed = [preprocess(text) for text in text_raw]
text_tokenized = [tokenize(text, model.wv.vocab) for text in text_preprocessed]
text_windowed = [list(zip(*[tokens[i:] for i in range(window_size)])) for tokens in text_tokenized]

vectors = np.zeros(shape=(len(data), 100))


iterations = 100
N = len(data)
n = int(N / iterations)
data.reset_index(drop=True, inplace=True)
for i, _ in data.iterrows():
    if i % n == 0:
        print(f"{i // n}/{iterations}")

    graph, indices = build_graph(text_windowed[i], model)
    ranks = build_ranks(graph, indices)

    ranks_dict = {word: ranks[i] for word, i in indices.items()}
    if len(ranks_dict) == 0:
        continue

    best = top_n(ranks_dict, 3)
    data.loc[i, "auto_keywords"] = ",".join(best)
    for word in best:
        vectors[i] += model.wv[word]
    vectors[i] /= len(best)

np.save("sentence_vectors", vectors)
data.to_csv("auto_keywords.csv", index=False)
